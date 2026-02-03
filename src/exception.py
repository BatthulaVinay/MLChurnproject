import sys
import os
from src.logger import logging


def error_message_detail(error_message, error_detail=None):
    if hasattr(error_detail, "exc_info"):
        _, _, exc_tb = error_detail.exc_info()
        return f"Exception in {exc_tb.tb_frame.f_code.co_filename}, line {exc_tb.tb_lineno}: {error_message}"
    else:
        return str(error_message)
    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    


       