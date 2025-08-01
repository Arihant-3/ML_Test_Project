import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    '''
    Generates a detailed error message.
    '''
    _, _, exc_tb = error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    error_message= "Error occurred in script: [{0}] at line number: [{1}] error message: [{2}]".format(
        filename, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    '''
    Custom exception class to handle exceptions with detailed error messages.
    '''
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
