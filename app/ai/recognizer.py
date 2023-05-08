from tempfile import SpooledTemporaryFile
from dataclasses import dataclass
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.cuda
from annoy import AnnoyIndex
from PIL import Image
import pickle


@dataclass
class RecognizeResult:
    user: str
    distance: float


class Recognizer:
    __slots__ = "device", "resnet", "mtcnn", "annoy", "person_name"

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.mtcnn = MTCNN(device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        self.annoy = AnnoyIndex(512, "euclidean")
        self.annoy.load("app/resources/annoy/index.ann")
        self.person_name = list()
        with open("app/resources/annoy/person_name", "rb") as person_name_file:
            self.person_name = pickle.load(file=person_name_file)

    def recognize(self, image_file: SpooledTemporaryFile) -> RecognizeResult:
        image = Image.open(image_file)
        cropped_face = self.mtcnn(image)
        if cropped_face is None:
            return None

        face_embedding = self.resnet(cropped_face.unsqueeze(0)).squeeze().detach().cpu()
        image.close()

        [nearest_face_embedding], [
            distance_to_nearest_face
        ] = self.annoy.get_nns_by_vector(face_embedding, 1, include_distances=True)
        if distance_to_nearest_face < 0.8:
            return RecognizeResult(
                self.person_name[nearest_face_embedding],
                distance=distance_to_nearest_face,
            )

        return RecognizeResult("Stranger", distance=distance_to_nearest_face)
