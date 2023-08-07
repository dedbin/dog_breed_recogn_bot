from model_for_is_dog import *


def is_dog_on_photo(img, outputs, class_names, white_list=['dog']):
    reses = []
    boxes, score, classes, nums = outputs
    boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        if white_list is not None and class_names[int(classes[i])] not in white_list:
            continue
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], score[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        reses.append(class_names[int(classes[i])])
    print(reses)
    return True if len(reses) > 0 else False

def prep_for_dog(img_path):
    image = img_path

    img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = preprocess_image(img, size)

    boxes, scores, classes, nums = yolo(img)
    img = cv2.imread(image)
    return is_dog_on_photo(img, (boxes, scores, classes, nums), class_names,)
