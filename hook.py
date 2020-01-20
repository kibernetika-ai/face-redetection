import cv2
from ml_serving.utils import helpers
import numpy as np


IMAGE_SIZE = 64


def prepare(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    return img


def process(inputs, ctx, **kwargs):
    original, is_video = helpers.load_image(inputs, 'input')
    image = original.copy()
    face_driver = ctx.drivers[0]
    facecheck_driver = ctx.drivers[1]
    input_name = list(facecheck_driver.inputs.keys())[0]

    boxes = get_boxes(face_driver, image)
    for box in boxes:
        box = box.astype(int)
        img = crop_by_box(image, box)

        prepared = prepare(img)
        prepared = np.expand_dims(prepared, axis=0)
        outputs = facecheck_driver.predict({input_name: prepared})
        cv2.rectangle(
            image,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 250, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(outputs['dense'][0][0]),
            (box[0], box[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 250, 0), 1, cv2.LINE_AA
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
