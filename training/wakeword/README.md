# ORION Wake Word Training

Este directorio contiene un notebook reproducible para entrenar un modelo wake word personalizado `Orion` con openWakeWord en Google Colab.

## Abrir en Colab

1. Sube o abre `training/wakeword/train_orion_colab.ipynb` en Google Colab.
2. Selecciona runtime:
   - Google Colab 2025.07
   - Python 3.11
   - GPU T4
3. Ejecuta las celdas en orden.

## Persistencia

El notebook guarda todo en Google Drive:

```text
/content/drive/MyDrive/orion_wakeword_training/
```

Si Colab se desconecta, vuelve a abrir el notebook, monta Drive y continúa. Las celdas son idempotentes: reutilizan repositorios, datasets, clips, features y modelos existentes cuando ya están en Drive.

## Flujo

1. Montar Drive.
2. Verificar Python/GPU.
3. Instalar dependencias necesarias.
4. Clonar `openWakeWord`.
5. Clonar `piper-sample-generator` en tag `v2.0.0`.
6. Aplicar parches de compatibilidad:
   - `torch.load(..., weights_only=False)`.
   - entrenamiento ONNX-only, sin TFLite.
7. Descargar modelos auxiliares y datasets:
   - `melspectrogram.onnx`
   - `embedding_model.onnx`
   - `en_US-libritts_r-medium.pt`
   - FMA
   - MIT RIR
   - `openwakeword_features_ACAV100M_2000_hrs_16bit.npy`
   - `validation_set_features.npy`
8. Crear `my_model_v2.yaml`.
9. Generar clips.
10. Verificar conteos:
   - `positive_train >= 1500`
   - `positive_test >= 1000`
   - `negative_train >= 1500`
   - `negative_test >= 1000`
11. Generar features.
12. Entrenar.
13. Verificar `orion_v2.onnx`.
14. Copiarlo a:

```text
/content/drive/MyDrive/orion_wakeword_training/exports/orion_v2.onnx
```

15. Descargarlo con `google.colab.files.download()`.

## Recuperar Después De Una Desconexión

Vuelve a ejecutar desde la primera celda. El notebook detecta lo existente en Drive y evita empezar desde cero. Si ya existe:

```text
/content/drive/MyDrive/orion_wakeword_training/exports/orion_v2.onnx
```

puedes saltar hasta la celda de descarga.

## Qué No Hace

- No entrena dentro de la `.venv` principal de ORION.
- No modifica `main.py`.
- No toca Whisper, Ollama, Policy Engine, ExecutionService ni el servicio wake word local.
- No usa AudioSet.
- No convierte a TFLite.
- No guarda `.onnx`, `.pt`, `.npy`, audios ni datasets en Git.
