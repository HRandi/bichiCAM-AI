# IMPORTER TOUS LES PLUGINS ET BIBLIOTHEQUES NECESSAIRES
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'
import time
import tensorflow as tf
import csv
import datetime
import statistics as stat
# Import Python OS tools
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import system.core.utils as utils
from system.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from system.core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import ConfigProto
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from system.deep_sort import preprocessing, nn_matching
from system.deep_sort.detection import Detection
from system.deep_sort.tracker import Tracker
from system.tools import generate_detections as gdet

st = datetime.datetime.now()
time_start = str(st.hour) + ':' + str(st.minute) + ':' + str(st.second)
# DEFINIR LES PARAMETRES UTILES POUR LA DETECTION, LE SUIVI ET LE COMPTAGE

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './system/lib/mr_sic_ind', 'chemin vers le modèle de référence')
flags.DEFINE_integer('size', 416, 'redimensionner les images pour')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './videos/test.mp4', 'chemin d''accès à la vidéo d''entrée ou défini sur 0 pour la webcam')
flags.DEFINE_string('output', None, 'chemin vers la sortie vidéo')
flags.DEFINE_string('output_format', 'XVID', 'codec utilisé lors de l''enregistrement de la vidéo dans un fichier')
flags.DEFINE_float('iou', 0.40, 'seuil de iou')  # 0.45
flags.DEFINE_float('score', 0.50, 'seuil de confiance')  # 0.50
flags.DEFINE_boolean('dont_show', False, 'ne pas afficher la sortie vidéo')


# LA FONCTION PRINCIPALE DU COMPTAGE
def main(_argv):
    # Définition des paramètres
    max_cosine_distance = 0.7
    nn_budget = None
    nms_max_overlap = 2.0

    # initialiser le tri approfondi (deep sort)
    model_filename = 'system/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculer la métrique de distance cosinus
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialiser le tracker
    tracker = Tracker(metric)

    # charger la configuration pour la détection d'objets
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # Autogenerate CSV filename based on video name
    # As .csv is appended below, we are getting the filename.extension and keeping just the filename
    # TODO: For future iterations, make this
    csv_name = os.path.basename(video_path).split('.')[0]

    # charger le modèle tflite si l'indicateur est défini
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # sinon charger le modèle enregistré standard Tensorflow
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    # commencer la capture/ouverture de la vidéo
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # préparer la vidéo à enregistrer localement si l'indicateur est défini
    if FLAGS.output:
        # par défaut VideoCapture renvoie float au lieu de int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    from _collections import deque
    pts = [deque(maxlen=30) for _ in range(10000000)]
    # Initialiser les variables de comptages
    counter = []
    counter_ind = []
    counterX = []
    counter_indX = []
    counterY = []
    counter_indY = []
    counterZ = []
    counter_indZ = []
    counterA = []
    counter_indA = []
    # pendant que la vidéo est en cours d’exécution
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Traitement terminé, essayez une autre vidéo !')
            break
        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        # exécuter des détections sur tflite si l'indicateur est défini
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # exécuter la détection à l'aide de yolov4 si l'indicateur est défini
            if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convertir les données en tableaux numpy et découper les éléments inutilisés
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # formater les boîtes englobantes à partir de ymin, xmin, ymax, xmax normalisés ---> xmin, ymin, largeur, hauteur
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # stocker toutes les prédictions dans un seul paramètre pour plus de simplicité lors de l'appel de fonctions
        pred_bbox = [bboxes, scores, classes, num_objects]
        # lire tous les noms de classe de config (dans coco.names)
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # par défaut, lister toutes les classes dans le fichier .names.
        allowed_classes = list(class_names.values())
        # classes autorisées personnalisées (décommentez la ligne ci-dessous pour personnaliser le tracker uniquement
        # pour sic) allowed_classes = ['sic'] boucle à travers les objets et utilise l'index de classe pour obtenir
        # le nom de la classe, autorise uniquement les classes dans la liste allowed_classes
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        # supprimer les détections qui ne sont pas dans allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encoder les détections yolo et alimenter le tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialiser la palette de couleurs
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        # exécuter une suppression non maximale
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Appeler le traqueur
        tracker.predict()
        tracker.update(detections)

        current_count = int(0)
        current_ind = int(0)
        current_countX = int(0)
        current_indX = int(0)
        current_countY = int(0)
        current_indY = int(0)
        current_countZ = int(0)
        current_indZ = int(0)
        current_countA = int(0)
        current_indA = int(0)
        # mettre à jour les bichiques identifiés
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            # dessiner les bbox à l'écran
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
            #               (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 0.5,
                        (255, 255, 255), 2)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            pts[track.track_id].append(center)

            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                # thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness=5)
                # cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)
            # Mettre à jour les lignes de comptages : 04/05/2021
            height, width, _ = frame.shape
            a = height - 100
            b = height - 280
            c = height - 280
            d = height - 450
            e = height - 450
            f = height - 630
            g = height - 630
            h = height - 810
            i = height - 810
            j = height - 990
            cv2.line(frame, (0, int(a)), (width, int(a)), (134, 134, 160),
                     thickness=2)  # Draw first line for counting area
            cv2.line(frame, (0, int(b)), (width, int(b)), (134, 134, 160),
                     thickness=2)  # Draw second line for counting area
            cv2.line(frame, (0, int(c)), (width, int(c)), (134, 134, 160),
                     thickness=2)  # Draw first line for counting area
            cv2.line(frame, (0, int(d)), (width, int(d)), (134, 134, 160),
                     thickness=2)  # Draw second line for counting area
            cv2.line(frame, (0, int(e)), (width, int(e)), (134, 134, 160),
                     thickness=2)  # Draw first line for counting area
            cv2.line(frame, (0, int(f)), (width, int(f)), (134, 134, 160),
                     thickness=2)  # Draw second line for counting area
            cv2.line(frame, (0, int(g)), (width, int(g)), (134, 134, 160),
                     thickness=2)  # Draw first line for counting area
            cv2.line(frame, (0, int(h)), (width, int(h)), (134, 134, 160),
                     thickness=2)  # Draw second line for counting area
            cv2.line(frame, (0, int(i)), (width, int(i)), (134, 134, 160),
                     thickness=2)  # Draw first line for counting area
            cv2.line(frame, (0, int(j)), (width, int(j)), (134, 134, 160),
                     thickness=2)  # Draw second line for counting area
            center_y = int(((bbox[1]) + (bbox[3])) / 2)
            # when the fish is in the current zone do : count fishes and add to previous total_count
            if int(a) >= center_y >= int(b):
                # if class_name == 'fish' or class_name == 'fish':
                if class_name == 'sic':
                    counter.append(int(track.track_id))
                    current_count += 1
                if class_name == 'ind':
                    counter_ind.append(int(track.track_id))
                    current_ind += 1
            if int(c) >= center_y >= int(d):
                # if class_name == 'fish' or class_name == 'fish':
                if class_name == 'sic':
                    counterX.append(int(track.track_id))
                    current_countX += 1
                if class_name == 'ind':
                    counter_indX.append(int(track.track_id))
                    current_indX += 1
            if int(e) >= center_y >= int(f):
                # if class_name == 'fish' or class_name == 'fish':
                if class_name == 'sic':
                    counterY.append(int(track.track_id))
                    current_countY += 1
                if class_name == 'sic':
                    counter_indY.append(int(track.track_id))
                    current_indY += 1
            if int(g) >= center_y >= int(h):
                # if class_name == 'fish' or class_name == 'fish':
                if class_name == 'sic':
                    counterZ.append(int(track.track_id))
                    current_countZ += 1
                if class_name == 'ind':
                    counter_indZ.append(int(track.track_id))
                    current_indZ += 1
            if int(i) >= center_y >= int(j):
                # if class_name == 'fish' or class_name == 'fish':
                if class_name == 'sic':
                    counterA.append(int(track.track_id))
                    current_countA += 1
                if class_name == 'ind':
                    counter_indA.append(int(track.track_id))
                    current_indA += 1

        total_count = len(set(counter))
        total_count_ind = len(set(counter_ind))
        total_esp = total_count + total_count_ind

        total_countX = len(set(counterX))
        total_count_indX = len(set(counter_indX))
        total_espX = total_countX + total_count_indX

        total_countY = len(set(counterY))
        total_count_indY = len(set(counter_indY))
        total_espY = total_countY + total_count_indY

        total_countZ = len(set(counterZ))
        total_count_indZ = len(set(counter_indZ))
        total_espZ = total_countZ + total_count_indZ

        total_countA = len(set(counterA))
        total_count_indA = len(set(counter_indA))
        total_espA = total_countA + total_count_indA

        list_calc = [total_esp, total_espX, total_espY, total_espZ, total_espA]
        list_calc_sic = [total_count, total_countX, total_countY, total_countZ, total_countA]
        list_calc_ind = [total_count_ind, total_count_ind, total_count_indY, total_count_indZ, total_count_indA]
        mediane_c = stat.median(list_calc)
        mediane_c_sic = stat.median(list_calc_sic)
        mediane_c_ind = stat.median(list_calc_ind)
        ecart = stat.pstdev(list_calc)
        ecart_sic = stat.pstdev(list_calc_sic)
        ecart_ind = stat.pstdev(list_calc_ind)
        ecart_type = float("{:.2f}".format(ecart))
        ecart_type_sic = float("{:.2f}".format(ecart_sic))
        ecart_type_ind = float("{:.2f}".format(ecart_ind))
        minimum = min(list_calc)
        minimum_sic = min(list_calc_sic)
        minimum_ind = min(list_calc_ind)
        maximum = max(list_calc)
        maximum_sic = max(list_calc_sic)
        maximum_ind = max(list_calc_ind)
        labelSic = "sic"
        labelInd = "ind"
        labelTotal = "total"

        # Draw black background rectangle
        cv2.putText(frame, "c_actuel: " + str(current_count), (10, 820), 0, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "c_cumule: " + str(total_count), (160, 820), 0, 0.7, (0, 0, 0), 2)
        # for ind
        cv2.putText(frame, "c_actuel: " + str(current_ind), (10, 855), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "c_cumule: " + str(total_count_ind), (160, 855), 0, 0.7, (0, 0, 255), 2)

        # ZONE DE COMPTAGE 2
        # for sic
        cv2.putText(frame, "c_actuel: " + str(current_countX), (10, 650), 0, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "c_cumule: " + str(total_countX), (160, 650), 0, 0.7, (0, 0, 0), 2)
        # for ind
        cv2.putText(frame, "c_actuel: " + str(current_indX), (10, 685), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "c_cumule: " + str(total_count_indX), (160, 685), 0, 0.7, (0, 0, 255), 2)

        # ZONE DE COMPTAGE 3
        # for sic
        cv2.putText(frame, "c_actuel: " + str(current_countY), (10, 465), 0, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "c_cumule: " + str(total_countY), (160, 465), 0, 0.7, (0, 0, 0), 2)
        # for ind
        cv2.putText(frame, "c_actuel: " + str(current_indY), (10, 500), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "c_cumule: " + str(total_count_indY), (160, 500), 0, 0.7, (0, 0, 255), 2)

        # ZONE DE COMPTAGE 4
        # for sic
        cv2.putText(frame, "c_actuel: " + str(current_countZ), (10, 295), 0, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "c_cumule: " + str(total_countZ), (160, 295), 0, 0.7, (0, 0, 0), 2)
        # for ind
        cv2.putText(frame, "c_actuel: " + str(current_indZ), (10, 330), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "c_cumule: " + str(total_count_indZ), (160, 330), 0, 0.7, (0, 0, 255), 2)

        # ZONE DE COMPTAGE 5
        # for sic
        cv2.putText(frame, "c_actuel: " + str(current_countA), (10, 120), 0, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "c_cumule: " + str(total_countA), (160, 120), 0, 0.7, (0, 0, 0), 2)
        # for ind
        cv2.putText(frame, "c_actuel: " + str(current_indA), (10, 150), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "c_cumule: " + str(total_count_indA), (160, 150), 0, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "mediane : ", (10, 1000), 0, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, str(mediane_c_sic), (100, 1000), 0, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, str(mediane_c_ind), (150, 1000), 0, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, str(mediane_c), (200, 1000), 0, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, "minimum: ", (10, 1025), 0, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "maximum: ", (10, 1050), 0, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, str(minimum_sic), (100, 1025), 0, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, str(maximum_sic), (100, 1050), 0, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, str(minimum_ind), (150, 1025), 0, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, str(maximum_ind), (150, 1050), 0, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, str(minimum), (200, 1025), 0, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, str(maximum), (200, 1050), 0, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "- stats", (10, 1070), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, str(labelSic), (100, 1070), 0, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, str(labelInd), (150, 1070), 0, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, str(labelTotal), (200, 1070), 0, 0.5, (255, 0, 0), 2)
        # Get date and time and save it inside a variable
        dt = datetime.datetime.now()
        chrono = str(dt.hour) + ':' + str(dt.minute) + ':' + str(dt.second)
        # Customize file output .csv with date
        date_traitement = str(dt.day) + "/" + str(dt.month) + "/" + str(dt.year)

        # Removing this as we generate a CSV filename based on the video name above
        # ref_video = "GH060025"  # A modifier selon la réf de la vidéo à analyser
        script = "20f_z5"
        modele = "sic_ind"
        label1 = "sic"
        label2 = "ind"
        # Put the date time over the video frame
        cv2.putText(frame, "Heure: " + str(chrono), (10, 30), 0, 0.75, (255, 255, 255), 2)
        # store data in csv with min max med etc...
        # TODO: Pour l'avenir, regardez l'utilisation de numpy ou pandas ici, on peut écrire nativement un CSV en une ligne
        with open(os.path.join('./src/csv/', f'{csv_name}_stats.csv'), 'w',
                  newline='') as csv_output:
            writer = csv.writer(csv_output)
            writer.writerow(
                [date_traitement, time_start, chrono, csv_name, script, modele, label1, total_count, total_countX,
                 total_countY, total_countZ, total_countA, mediane_c_sic, ecart_type_sic, minimum_sic, maximum_sic])
            writer.writerow(
                [date_traitement, time_start, chrono, csv_name, script, modele, label2, total_count_ind,
                 total_count_ind, total_count_indY, total_count_indZ, total_count_indA, mediane_c_ind,
                 ecart_type_ind,
                 minimum_ind, maximum_ind])
        # calculer les images par seconde des détections en cours
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Bichicam", flags=0)
        cv2.resizeWindow("Bichicam", 1080, 800)

        if not FLAGS.dont_show:
            cv2.imshow("Bichicam", result)
        # si l'indicateur de sortie est défini, enregistrez le fichier vidéo
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
