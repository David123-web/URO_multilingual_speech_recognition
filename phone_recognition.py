from flowmason import conduct, MapReduceStep, SingletonStep, load_artifact # TODO: install this package from me: https://github.com/smfsamir/flowmason

# from allosaurus.app import read_recognizer
import panphon.distance # https://github.com/dmort27/panphon

# TODO: you'll have to install these packages. Let me know if there's any trouble here; you should be able to do `pip install` for all of them
import polars as pl
import torch
import soundfile as sf
import os
import ipdb

from typing import List
from collections import OrderedDict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC 

HF_CACHE_DIR="[FILL THIS IN]"
SCRATCH_DIR="[FILL THIS IN]"

def step_generate_pfer(prediction_frame: pl.DataFrame, 
                                  **kwargs):
    feat_edit_distance = panphon.distance.Distance().feature_edit_distance
    # currently, the predictions have spaces between segments. Remove the spaces
    # in the predictions column
    prediction_frame = prediction_frame.with_columns([
        pl.col('predictions').map_elements(lambda x: x.replace(' ', '')).alias('predictions')
    ])
    prediction_frame = prediction_frame.with_columns([
        pl.struct(['predictions', 'transcripts']).map_elements(
            lambda x: feat_edit_distance(x['predictions'], x['transcripts'])).alias('pfer')
    ])
    # group by the language code and compute the average pfer
    return prediction_frame

def step_generate_predictions_notre_dame(**kwargs) -> str:
    model = Wav2Vec2ForCTC.from_pretrained("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", cache_dir=HF_CACHE_DIR)
    processor = Wav2Vec2Processor.from_pretrained("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", cache_dir=HF_CACHE_DIR)
    predictions = []
    transcripts = []
    for datapoint in get_get_data_iterator(): # TODO: you have to implement this function to iterate over the arctic samples
        audio, txt = datapoint # TODO: the iterator should returns tuples of the audio array and the transcript
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0]
        predictions.append(prediction)
        transcripts.append(txt)
    return pl.DataFrame({
        "predictions": predictions,
        "transcripts": transcripts
    })

if __name__ == "__main__":
    step_dict = OrderedDict()

    notre_dame_steps = OrderedDict()
    notre_dame_steps['step_generate_preds_notre_dame'] = SingletonStep(step_generate_predictions_notre_dame, {
        "version": "001"
    })
    
    step_dict['step_generate_pfer_notre_dame'] = SingletonStep(step_generate_pfer, {
        'version': '001', 
        'prediction_frame': 'map_step_generate_notredame_preds'
    })
    step_dict['map_step_generate_notredame_preds'] = MapReduceStep(
        notre_dame_steps, {},
         {
             "version": "001"
         },
         pl.concat
    )
    metadata = conduct(os.path.join(SCRATCH_DIR, "[FILL THIS IN]"), step_dict, "[FILL THIS IN]")
    ipdb.set_trace()
