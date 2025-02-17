# -*- coding: utf-8 -*-

from pydantic import BaseModel
from openai import OpenAI
from typing import List


class Step(BaseModel):
    title: str
    description: str

class control_response(BaseModel):
    marker_id: int
    linear_x: float
    angular_z: float

class Multi_Robot_Control(BaseModel):
    steps: List[Step]
    final_response: List[control_response]


