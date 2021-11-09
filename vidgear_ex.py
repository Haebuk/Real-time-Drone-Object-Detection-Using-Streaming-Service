# import required libraries
from vidgear.gears import CamGear
import cv2
import pafy

# set desired quality as 720p
options = {"STREAM_RESOLUTION": "720p"}

# Add any desire Video URL as input source
# for e.g https://vimeo.com/151666798
# and enable Stream Mode (`stream_mode = True`)
url = "https://www.youtube.com/watch?v=uyB_ZYin4ew"
stream = CamGear(
    source=url,
    stream_mode=True,
    logging=True,
    **options
).start()

video = pafy.new(url).getbest(preftype="mp4") # pafy video
cap = cv2.VideoCapture(video.url) # cv2 video

# loop over
while True:

    # read frames from stream
    frame = stream.read()
    grabbed, frame2 = cap.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output-vidgear", frame)
    cv2.imshow('Output-pafy', frame2)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()