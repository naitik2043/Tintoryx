# Third-Party Model and Software Notices

## DDColor-tiny

Tintoryx's recommended photograph-colorization backend uses a quantized ONNX
export of **DDColor-tiny**, based on the DDColor project presented at ICCV 2023.

- Original implementation: https://github.com/piddnad/DDColor
- ONNX model repository: https://huggingface.co/toolsmonk/ddcolor-tiny
- Pinned ONNX repository revision: `c7d348d`
- Artifact: `ddcolor_tiny_512_int8.onnx` (112,723,138 bytes)
- SHA-256: `e983347272b8e218edc974a0af2bd66f53b73212abb800e103c64806e4c14a14`
- Model license: Apache License 2.0, as stated by the model card.

The ONNX model card describes the artifact as a DDColor-tiny export with
MatMul and Gemm weights dynamically quantized to uint8. It reports a mean
absolute difference of approximately 1.1 Lab chroma units versus its FP32
export on a test image. Tintoryx verifies the downloaded artifact against its
published SHA-256 before loading it.

The model file is downloaded on first use and cached locally; it is not
committed to this repository.

The MIT license in the repository root applies to Tintoryx's own code. It does
not replace or override the separate Apache-2.0 license for the DDColor model
and upstream implementation. Review the original model card and upstream
project license before redistributing model weights or deploying commercially.

## DDColor citation

```bibtex
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran
          and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Runtime components

Tintoryx uses ONNX Runtime for model execution, OpenCV for CIE Lab color
conversion, Hugging Face Hub for first-use model retrieval, and Streamlit for
the web interface. Their package licenses and notices remain with their
respective projects.
