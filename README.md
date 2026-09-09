# image-classifier-model

A dense neural network that classifies clothing images using the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset
(28x28 px, grayscale, 10 classes). The trained model is exported to H5 and,
optionally, converted to a format that
[TensorFlow.js](https://www.tensorflow.org/js) can load in the browser.

## Requirements

- Python 3.10
- macOS / Linux (on Windows the venv activation commands differ)

## Layout

| File | Purpose |
|---|---|
| `main.py` | Loads the data, trains the network, and exports `exported_model.h5` |
| `convert.sh` | Converts `exported_model.h5` to a TensorFlow.js model in `tfjs_model/` |
| `index.html` | Browser demo: upload a clothing image and let the model classify it |
| `requirements.txt` | Dependencies for **training** (TensorFlow 2.19) |
| `requirements-tfjs.txt` | Dependencies for **converting** the model to TensorFlow.js — installed in a separate venv |
| `data/` | Dataset downloaded by `tfds.load` (git-ignored, regenerated automatically) |

## 1. Train the model

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The first run downloads Fashion-MNIST (~36 MB) into `data/`. Later runs read it from there.

When it finishes it produces `exported_model.h5` in the project root.

## 2. Convert to TensorFlow.js (optional)

`tensorflowjs` pulls in `tensorflow-decision-forests`, which requires a different
TensorFlow version than the one used for training. That's why it is **not** in
`requirements.txt` and is installed in a separate virtual environment:

```bash
deactivate                          # leave the training venv if it's active
python3 -m venv .venv-tfjs
source .venv-tfjs/bin/activate
pip install -r requirements-tfjs.txt
```

With the model already trained (`exported_model.h5`) and the `.venv-tfjs` environment
active, run the conversion script:

```bash
./convert.sh
```

It runs `tensorflowjs_converter` and creates `tfjs_model/` with `model.json` and the
`.bin` weight files, ready to load with `tf.loadLayersModel()` in the browser.

When done:

```bash
deactivate
```

## 3. Try the model in the browser

`index.html` loads `tfjs_model/` with TensorFlow.js and lets you upload a clothing
image. It must be served over HTTP (opening the file directly won't let the
browser fetch `tfjs_model/model.json`).

From the project root (any environment, no venv needed):

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>, pick an image, and click **Predict**. The image
is downscaled to 28x28 grayscale. Keep **Invert colors** checked for a normal photo
(dark garment on a light background) so it matches how Fashion-MNIST is stored
(light garment on a dark background); the "What the model sees" preview shows the
result. Accuracy on real photos is limited — the model was trained on Fashion-MNIST,
not arbitrary images.

## Notes

- `main.py` sets `TF_USE_LEGACY_KERAS=1` (and `requirements.txt` includes `tf-keras`)
  so the model is built with Keras 2. Keras 3 saves the input layer in a format that
  the TensorFlow.js converter does not translate, which breaks `tf.loadLayersModel()`.
- The virtual environments (`.venv/`, `.venv-tfjs/`) and `data/` are git-ignored.
- The exported model (`exported_model.h5`) and `tfjs_model/` are committed, so the browser model can be loaded without retraining.
- To reproduce the exact training environment on another machine, run `pip install -r requirements.txt`.
