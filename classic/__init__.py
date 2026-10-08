"""classic engine — pure-CV algorithms ported from tools mcsearch/.

ColorModelV2  : HSV palette -> LAB smoothed histogram -> 81-dim color features
                (source: model-infer-api vcgImageAI/subProjects/mcsearch/colorModel.py)
ColorPalette  : faiss.Kmeans dominant colors -> [(hex, weight)] (source: same dir,
                imageColorPalette.py / ColorPaletteModel — identical implementation)
OPQQuantizer  : OPQ rotation + PQ encode of the 81-dim features -> tools
                opqCode string 'code_0 code_1 code_2' (source:
                vcgImageAI/subProjects/featuresRetrive/quantizers.py,
                OPQFeaturesQuantizer.quantize(needTransformer=True))

PQ codebooks: shipped verbatim from the tools container
(/root/.cache/soujpg/models/ytuHighEnd-color_81-es_opq-10_3/{opq,pq}_3_10.model,
md5 2d515c4364a1b6dc7961bbe42ce872b3 / 3428f147a18f88652fc1a6db95b9309f) —
NEVER retrained: ES stored opqCode must stay in the same code space.
"""
