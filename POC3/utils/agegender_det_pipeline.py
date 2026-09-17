import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from utils.mivolo_model import MiVOLOPredictor


# ============================================================
# PyTorch / Ultralytics safe loading

torch.serialization.add_safe_globals([ DetectionModel ])


class AgeGenderPipeline:

    def __init__(
        self,
        face_model="buffalo_l",
        person_model="yolov8n.pt",
        mivolo_config=( "/home/webclues-nidhi/mYpY/RnD/Age_detectoion/models/config.json" ),
        mivolo_weights=( "/home/webclues-nidhi/mYpY/RnD/Age_detectoion/models/model.safetensors" ),
        device="cpu",
        face_det_size=(640, 640),
        person_conf=0.25,
        face_conf=0.40,
        person_iou=0.45,
    ):
        if device is None:
            self.device = ( "cuda" if torch.cuda.is_available() else "cpu" )
        else:
            self.device = device

        print( "Age / Gender Pipeline" )
        print( "Device:", self.device, )

        # ----------------------------------------------------
        # Buffalo

        print( "\nLoading InsightFace Buffalo..." )

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.startswith("cuda") else ["CPUExecutionProvider"])

        self.face_app = FaceAnalysis( name=face_model, providers=providers )
        ctx_id = ( 0 if self.device.startswith("cuda") else -1 )
        self.face_app.prepare( ctx_id=ctx_id, det_size=face_det_size )
        self.face_conf = float( face_conf )

        print( "Buffalo loaded." )

        # ----------------------------------------------------
        # YOLO

        print( "\nLoading YOLO person detector..." )
        print( "Model:", person_model, )
        
        self.person_detector = YOLO( person_model )
        self.yolo_device = ( self.device if self.device.startswith("cuda") else "cpu" )
        self.person_conf = float( person_conf )
        self.person_iou = float( person_iou )

        print( "YOLO loaded." )

        # ----------------------------------------------------
        # MiVOLO

        print( "\nLoading MiVOLO..." )

        self.mivolo = MiVOLOPredictor(
            config_path=mivolo_config,
            weights_path=mivolo_weights,
            device=self.device,
        )

        print( "MiVOLO loaded." )
        print("Pipeline ready")

    # ========================================================
    # BBOX

    @staticmethod
    def clip_bbox( bbox, image_shape, ):

        h, w = image_shape[:2]
        x1, y1, x2, y2 = [ int(round(v)) for v in bbox ]
        x1 = max( 0, min(x1, w - 1) )
        y1 = max( 0, min(y1, h - 1) )
        x2 = max( 0, min(x2, w) )
        y2 = max( 0, min(y2, h) )

        return [ x1, y1, x2, y2 ]

    @staticmethod
    def bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return float( max(0, x2 - x1) * max(0, y2 - y1) )

    @staticmethod
    def bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0, )

    @staticmethod
    def bbox_iou( box_a, box_b, ):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max( ax1, bx1, )
        iy1 = max( ay1, by1, )
        ix2 = min( ax2, bx2, )
        iy2 = min( ay2, by2, )
        iw = max( 0, ix2 - ix1, )
        ih = max( 0, iy2 - iy1, )
        intersection = ( iw * ih )

        if intersection <= 0: return 0.0

        area_a = ( AgeGenderPipeline .bbox_area(box_a) )
        area_b = ( AgeGenderPipeline .bbox_area(box_b) )
        union = ( area_a + area_b - intersection )

        if union <= 0: return 0.0

        return ( intersection / union )

    @staticmethod
    def face_inside_person( face_bbox, person_bbox ):

        fx1, fy1, fx2, fy2 = ( face_bbox )
        px1, py1, px2, py2 = ( person_bbox )
        ix1 = max( fx1, px1 )
        iy1 = max( fy1, py1 )
        ix2 = min( fx2, px2 )
        iy2 = min( fy2, py2 )

        intersection = ( max(0, ix2 - ix1) * max(0, iy2 - iy1) )
        face_area = ( AgeGenderPipeline .bbox_area(face_bbox) )
        if face_area <= 0: return 0.0

        return ( intersection / face_area )

    # ========================================================
    # FACE FALLBACK CROP

    @staticmethod
    def expand_bbox( image, bbox, scale=2.5, ):

        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        cx = ( x1 + x2 ) / 2.0
        cy = ( y1 + y2 ) / 2.0
        bw = ( x2 - x1 )
        bh = ( y2 - y1 )
        new_w = ( bw * scale )
        new_h = ( bh * scale )
        nx1 = int( cx - new_w / 2 )
        ny1 = int( cy - new_h / 2 )
        nx2 = int( cx + new_w / 2 )
        ny2 = int( cy + new_h / 2 )

        return [ max(0, nx1), max(0, ny1), min(w, nx2), min(h, ny2) ]

    # ========================================================
    # CROP

    @staticmethod
    def crop_image( image, bbox ):
        x1, y1, x2, y2 = [ int(v) for v in bbox]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return None
        return crop

    # ========================================================
    # PERSON DETECTION

    def detect_persons( self, image ):

        results = (
            self.person_detector.predict(
                source=image,
                conf=self.person_conf,
                iou=self.person_iou,
                classes=[0],
                device=self.yolo_device,
                verbose=False,
            )
        )
        persons = []
        if not results: return persons
        result = results[0]
        if result.boxes is None: return persons

        boxes = result.boxes
        xyxy = ( boxes.xyxy .detach() .cpu() .numpy() )
        confs = ( boxes.conf .detach() .cpu() .numpy() )

        for bbox, confidence in zip( xyxy, confs ):
            bbox = self.clip_bbox( bbox, image.shape )
            x1, y1, x2, y2 = bbox
            if ( x2 <= x1 or y2 <= y1 ):
                continue

            persons.append({ "bbox": bbox, "confidence": float( confidence ) })

        return persons

    # ========================================================
    # MATCH FACE TO PERSON

    def match_face_to_person( self, face_bbox, persons ):

        if not persons: return None

        fx, fy = ( self.bbox_center( face_bbox ) )
        
        candidates = []

        for person in persons:
            person_bbox = ( person["bbox"] )
            containment = (self.face_inside_person( face_bbox, person_bbox ))
            iou = self.bbox_iou( face_bbox, person_bbox )
            px, py = ( self.bbox_center(person_bbox) )
            pw = max( 1.0, person_bbox[2] - person_bbox[0] )
            ph = max( 1.0, person_bbox[3] - person_bbox[1] )
            center_distance = ( abs(fx - px) / pw + abs(fy - py) / ph )
            score = ( containment * 10.0 + iou * 3.0 - center_distance )
            candidates.append(( score, containment, iou, person ))

        candidates.sort( key=lambda x: x[0], reverse=True )

        ( score, containment, iou, best_person, ) = candidates[0]

        if containment >= 0.50: return best_person
        fx, fy = ( self.bbox_center(face_bbox) )

        px1, py1, px2, py2 = ( best_person["bbox"] )

        if ( px1 <= fx <= px2 and py1 <= fy <= py2 ): return best_person
        if iou >= 0.10: return best_person

        return None

    # ========================================================
    # PROCESS IMAGE

    def process_image( self, original_image, draw=False, return_detections=True ):

        if original_image is None:
            raise ValueError( "original_image is None" )

        if not isinstance( original_image, np.ndarray ):
            raise TypeError( "original_image must be " "a NumPy array" )

        # ----------------------------------------------------
        # PERSONS
        print("PERSON DETECTION")
        
        persons = ( self.detect_persons( original_image ) )

        print("Detected persons:", len(persons))
        # ----------------------------------------------------
        # FACES
        print("FACE DETECTION")

        faces = self.face_app.get( original_image )

        print("Detected faces:",len(faces))
        # ----------------------------------------------------
        # DRAW

        annotated = ( original_image.copy() )
        if draw:
            for i, person in enumerate( persons ):
                x1, y1, x2, y2 = ( person["bbox"] )
                cv2.rectangle( annotated, (x1, y1), (x2, y2), (255, 0, 0), 2 )
                cv2.putText( annotated, ( f"person {i} " f"{person['confidence']:.2f}" ), ( x1, max( 20, y1 - 5, ), ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, )

        output_results = []
        # ----------------------------------------------------
        # EACH FACE

        for face_index, face in enumerate( faces ):
            bbox = (face.bbox.astype(float))
            bbox = self.clip_bbox(bbox, original_image.shape)
            x1, y1, x2, y2 = bbox
            face_width = (x2 - x1)
            face_height = (y2 - y1)
            if (face_width <= 0 or face_height <= 0):
                continue

            face_score = getattr( face, "det_score", None )

            if face_score is not None:
                face_score = float(face_score)
                if (face_score < self.face_conf):
                    continue

            # ------------------------------------------------
            # MATCH PERSON
            matched_person = ( self.match_face_to_person( bbox, persons ) )

            # ------------------------------------------------
            # FACE CROP
            face_crop = ( self.crop_image( original_image, bbox ) )

            if face_crop is None:
                continue
            # ------------------------------------------------
            # PERSON CROP

            used_person_detector = False
            fallback_used = False

            if matched_person is not None:
                person_bbox = ( matched_person["bbox"] )
                person_crop = ( self.crop_image( original_image, person_bbox ) )

                if person_crop is not None:
                    used_person_detector = True
                else:
                    person_crop = None
            else:
                person_bbox = None
                person_crop = None

            # ------------------------------------------------
            # FALLBACK

            if person_crop is None:
                person_bbox = ( self.expand_bbox( original_image, bbox, scale=2.5 ) )
                person_crop = ( self.crop_image( original_image, person_bbox ) )
                fallback_used = True

            if person_crop is None:
                print(f"[Face {face_index}] Person crop failed.")
                continue

            # ------------------------------------------------
            # DEBUG
            print(f"FACE (index) {face_index}")
            print("Face bbox:", bbox)
            print("Face crop:", face_crop.shape)
            print("Person crop:", person_crop.shape)
            print("Face confidence:", face_score)
            print("Matched person:", matched_person is not None)
            print("Using YOLO person crop:", used_person_detector)
            print("Using fallback:", fallback_used)

            # ------------------------------------------------
            # MiVOLO

            prediction = ( self.mivolo.predict( face_image=face_crop, person_image=person_crop ) )

            print( "MiVOLO:", prediction )

            # ------------------------------------------------
            # RESULT

            result = {
                "face_index": face_index,
                "bbox": [ int(x1), int(y1), int(x2), int(y2) ],
                "age": prediction["age"],
                "gender": prediction["gender"],
                "gender_confidence": prediction[ "gender_confidence" ],
                "raw_logits": prediction[ "raw_logits" ],
                "face_confidence": ( round( face_score, 3 ) if face_score is not None else None ),
                "person_bbox": ( [ int(v) for v in person_bbox ] if person_bbox is not None else None ),
                "person_confidence": ( round(matched_person[ "confidence" ], 3) if matched_person is not None else None ),
                "used_person_detector": used_person_detector,
                "used_fallback_person_crop": fallback_used,
            }

            output_results.append( result )
            
            # ------------------------------------------------
            # DRAW FACE

            if draw: 
                cv2.rectangle( annotated, (x1, y1), (x2, y2), (0, 255, 0), 2, )
                label = (f"{prediction['age']:.1f}y {prediction['gender']} {prediction['gender_confidence']:.2f}")
                cv2.putText( annotated, label, ( x1, max( 25, y1 - 10, ), ), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2 )
                # Person bbox
                if matched_person is not None:
                    px1, py1, px2, py2 = ( matched_person[ "bbox" ] )
                    cv2.rectangle( annotated, (px1, py1), (px2, py2), (255, 0, 255), 2 )

        # ----------------------------------------------------        
        if draw:
            return { "results": output_results, "image": annotated }

        return output_results