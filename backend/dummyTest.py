"""
This module contains dummy functions for testing fronted-backend integration. Note that this module will be removed before project completion.
"""

STEP_IMAGE_PATHS = [
    "/static/steps/step1.png",
    "/static/steps/step2.png",
    "/static/steps/step3.png",
    "/static/steps/step4.png",
    "/static/steps/step5.png",
    "/static/steps/step6.png",
    "/static/steps/step7.png",
    "/static/steps/step8.png"
    ]

index: int = 0


def test_next():
    global index
    if index < len(STEP_IMAGE_PATHS) - 1:
        index += 1

    message: str = "NEXT button (backend) has been pressed"
    print(message, index)
    return STEP_IMAGE_PATHS[index]

def test_prev():
    global index
    if index > 0:
        index -= 1

    message: str = "PREV button (backend) has been pressed"
    print(message, index)
    return STEP_IMAGE_PATHS[index]
