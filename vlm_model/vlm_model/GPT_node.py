# -*- coding: utf-8 -*-

# ROS related
import rclpy
from rclpy.node import Node
from msgs_interfaces.srv import GPT

# LLM related
import json
import os
import time
from openai import OpenAI
from .openai_config import OpenAIConfig
from .response_format import Multi_Robot_Control

import cv2
from cv_bridge import CvBridge
import base64

# Global Initialization
config = OpenAIConfig()
client = OpenAI(
    api_key=config.openai_api_key
    
)

class GPTNode(Node):
    def __init__(self):
        super().__init__("GPT_node")

        # Server for GPT response: (Poses, Image) -> Twist
        self.function_call_server = self.create_service(
            GPT, "GPT_service", self.gpt_callback
        )
        
        self.bridge = CvBridge()

        # Initialize chat history
        self.start_timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.chat_history_file = os.path.join(
            config.chat_history_path, f"multi_robot_{self.start_timestamp}.json"
        )
        self.write_chat_history_to_json()


    ##################################### CHATGPT RESPONSE ############################################
    def gpt_callback(self, request, response):

        self.add_message_to_history("user", content=request.poses_text, image=request.image) 

        gpt_response = self.generate_gpt_response(config.chat_history)
        content, final_response = self.format_response_information(gpt_response)
        
        response_text = json.dumps(final_response)

        for marker in final_response:
            if marker['marker_id'] == 10:
                response.response_linearx1 = marker['linear_x']
                response.response_angularz1 = marker['angular_z']
            elif marker['marker_id'] == 20:
                response.response_linearx2 = marker['linear_x']
                response.response_angularz2 = marker['angular_z']
            # elif marker['marker_id'] == 30:
            #     response.response_linearx3 = marker['linear_x']
            #     response.response_angularz3 = marker['angular_z']

        self.add_message_to_history(
            role="assistant",
            content=response_text,
            image=None
        )
        self.get_logger().info(f"Response: {response}") 

        return response
        

    def generate_gpt_response(self, chat_history):
        response = client.beta.chat.completions.parse(
            model=config.openai_model,
            messages=chat_history,
            response_format=Multi_Robot_Control
            # functions=config.response_functions_list,
            # function_call="auto"
        )
        # self.get_logger().info(f"OpenAI response: {response}")
        return response


    def format_response_information(self, gpt_response):

        message = gpt_response.choices[0].message
        content = message.content.strip("\n").strip()
        self.get_logger().info(f"MESSAGE: {content}")

        content = json.loads(content)
        steps = content["steps"]
        final_response = content["final_response"]
        # steps = gpt_response.choices[0].message.parsed.steps
        # final_response = gpt_response.choices[0].message.parsed.final_response

        return content, final_response
        

    ####################################### HELPER FUNCTIONS ############################################

    def add_message_to_history(self, role, content="null", image=None):

        if role == "user":
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            _, buffer = cv2.imencode('.png', cv_image)

            encoded_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
            message_element_object = {
                "role": role,
                "content": [
                            {"type": "text", "text": content},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                            },
                ],
            }
        else:
            message_element_object = {
                "role": role,
                "content": [
                            {"type": "text", "text": content},
                ],
            }
        
        config.chat_history.append(message_element_object)
        self.write_chat_history_to_json()

        if len(config.chat_history) > config.chat_history_max_length: 
            self.get_logger().info(
                f"Chat history is too long, popping out the oldest message: {config.chat_history[0]}"
            )
            config.chat_history.pop(0)

        return config.chat_history


    def write_chat_history_to_json(self):
        try:
            # Converting chat history to JSON string
            json_data = json.dumps(config.chat_history)

            # Writing JSON to file
            with open(self.chat_history_file, "w", encoding="utf-8") as file:
                file.write(json_data)

            return True

        except IOError as error:
            self.get_logger().error(f"Error writing chat history to JSON: {error}")
            return False

def main(args=None):
    rclpy.init(args=args)
    chatgpt = GPTNode()
    rclpy.spin(chatgpt)
    rclpy.shutdown()


if __name__ == "__main__":
    main()