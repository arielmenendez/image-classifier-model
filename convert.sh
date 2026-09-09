#!/bin/sh
# Convert exported_model.h5 to a TensorFlow.js model in tfjs_model/.
# Run inside the .venv-tfjs environment (see README section 2).
set -e

if [ ! -f exported_model.h5 ]; then
  echo "exported_model.h5 not found. Train the model first: python main.py" >&2
  exit 1
fi

if ! command -v tensorflowjs_converter >/dev/null 2>&1; then
  echo "tensorflowjs_converter not found. Activate the .venv-tfjs environment first:" >&2
  echo "  source .venv-tfjs/bin/activate" >&2
  exit 1
fi

tensorflowjs_converter --input_format keras exported_model.h5 tfjs_model
echo "Done. Output in tfjs_model/"
