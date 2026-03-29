from intent.encoder import IntentEncoder
from intent.quantizer import Quantizer
from intent.sparsifier import Sparsifier
from intent.bandwidth import TokenBucket
from presence.reconstructor import PresenceReconstructor
from presence.renderer import FaceRenderer
from metrics.logger import MetricsLogger

import time

FPS = 15
BANDWIDTH_KBPS = 5.0

def main():
    encoder = IntentEncoder()
    quantizer = Quantizer(bits=6)
    sparsifier = Sparsifier(mode="relative")
    bucket = TokenBucket(rate_kbps=BANDWIDTH_KBPS)
    reconstructor = PresenceReconstructor()
    renderer = FaceRenderer()
    logger = MetricsLogger()

    prev_intent = None

    while True:
        intent = encoder.capture()

        delta = intent if prev_intent is None else intent - prev_intent
        q = quantizer.encode(delta)
        sparse, K = sparsifier.apply(q)

        sent = bucket.allow(sparse.nbytes * 8)
        if sent:
            recon = reconstructor.update(sparse)
            error = reconstructor.error()
        else:
            recon = reconstructor.hold()
            error = reconstructor.error()

        renderer.draw(recon)
        logger.log(error, bucket.kbps(), sent, K)

        prev_intent = intent
        time.sleep(1 / FPS)

if __name__ == "__main__":
    main()
