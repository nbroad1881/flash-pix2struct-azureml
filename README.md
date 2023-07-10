# flash-pix2struct-azureml

This repo contains the code to run pix2struct in Azure ML using flash attention. There is only flash attention in the encoder, because the decoder has an attention bias mechanism that isn't compatible.  Flash attention can help save memory when training with a large number of patches (2k+). 

The existing code uses a public dataset, so you'll likely want to point it to a local directory of files. I do not do any processing on the dataset: the model will try to generate json code. Some other approaches will add special tokens that delimit the generated text to make it easier to parse.

This uses the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) from April of 2023 (nvcr.io/nvidia/pytorch:23.04-py3). To use prompts for the model, the following was added to the Dockerfile.

```
RUN apt-get update && apt-get -y install libfreetype6-dev
RUN pip uninstall -y pillow && \
    pip install --no-cache-dir pillow
```

The default font file is also included.


Running this is a simple as executing each cell in the notebook. Everything else should be self-explanatory, but if you have a question please open an issue.