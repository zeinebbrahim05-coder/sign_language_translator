def detect_gesture(fingers):

    if fingers == [0,1,0,0,0]:
        return "ONE"

    elif fingers == [0,1,1,0,0]:
        return "TWO"

    elif fingers == [0,0,0,0,0]:
        return "FIST"

    else:
        return "UNKNOWN"