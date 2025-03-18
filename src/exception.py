import sys


def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error message: {str(error)}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name: [{file_name}]\nLine number [{exc_tb.tb_lineno}]\nError message [{str(error)}]"

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)

        # Handle both Exception objects and strings
        if isinstance(error_message, Exception):
            self.error_message = error_message_detail(error_message, error_detail=error_detail)
        else:
            self.error_message = error_message_detail(Exception(error_message), error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
