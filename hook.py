import math

import cv2
from ml_serving.utils import helpers
import numpy as np
from scipy import special


idx_tensor = np.arange(0, 101)


def process(inputs, ctx, **kwargs):
    original, is_video = helpers.load_image(inputs, 'input')
    image = original.copy()
    face_driver = ctx.drivers[0]
    age_driver = ctx.drivers[1]

    boxes = get_boxes(face_driver, image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        box = box.astype(int)
        margin = int(((box[2] - box[0]) / 2 + (box[3] - box[1]) / 2) * 0.4)
        img = crop_by_box(image, box, margin=margin)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        prepared = img.transpose([2, 0, 1])
        prepared = np.expand_dims(prepared, axis=0)
        outputs = age_driver.predict({'input.1': prepared.astype(np.float)})
        output = special.softmax(list(outputs.values())[0])
        predicted_ages = (output * idx_tensor).sum(axis=-1)
        cv2.putText(
            image,
            str(int(round(predicted_ages[0]))),
            (box[0], box[1]),
            font, 1.5, color=(0, 0, 250),
            thickness=2, lineType=cv2.LINE_AA
        )

    if is_video:
        output = image
    else:
        _, buf = cv2.imencode('.jpg', image[:, :, ::-1])
        output = buf.tostring()

    return {'output': output}


def get_boxes(face_driver, frame, threshold=0.5, offset=(0, 0)):
    input_name, input_shape = list(face_driver.inputs.items())[0]
    output_name = list(face_driver.outputs)[0]
    inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = face_driver.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    xmin = boxes[:, 0] * frame.shape[1] + offset[0]
    xmax = boxes[:, 2] * frame.shape[1] + offset[0]
    ymin = boxes[:, 1] * frame.shape[0] + offset[1]
    ymax = boxes[:, 3] * frame.shape[0] + offset[1]
    xmin[xmin < 0] = 0
    xmax[xmax > frame.shape[1]] = frame.shape[1]
    ymin[ymin < 0] = 0
    ymax[ymax > frame.shape[0]] = frame.shape[0]

    boxes[:, 0] = xmin
    boxes[:, 2] = xmax
    boxes[:, 1] = ymin
    boxes[:, 3] = ymax
    return boxes


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box, margin=0):
    h = (box[3] - box[1])
    w = (box[2] - box[0])
    ymin = int(max([box[1] - h * margin, 0]))
    ymax = int(min([box[3] + h * margin, img.shape[0]]))
    xmin = int(max([box[0] - w * margin, 0]))
    xmax = int(min([box[2] + w * margin, img.shape[1]]))
    return img[ymin:ymax, xmin:xmax]


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
