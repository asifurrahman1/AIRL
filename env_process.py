import cv2
def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state