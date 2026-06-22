# ORION Wake Word Training

Este directorio contiene el notebook reproducible para entrenar el wake word personalizado `Orion` con openWakeWord en Google Colab.

## Abrir En Colab

1. Abre `training/wakeword/train_orion_colab.ipynb` en Google Colab.
2. Selecciona el runtime recomendado:
   - Google Colab 2025.07
   - Python 3.11
   - GPU T4
3. Ejecuta las celdas en orden.

## Carpeta Persistente

Todo el progreso reutilizable vive en Google Drive:

```text
/content/drive/MyDrive/orion_wakeword_training/
```

El notebook reutiliza los recursos existentes cuando ya son validos:

```text
features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
features/validation_set_features.npy
datasets/fma/
datasets/mit_rirs/
models/piper/en_US-libritts_r-medium.pt
models/piper/en_US-libritts_r-medium.pt.json
training_output/orion_v2/
my_model_v2.yaml
```

`piper-sample-generator` no se guarda en Drive. Se clona de forma temporal en:

```text
/content/piper-sample-generator-v2
```

## Flujo Del Notebook

1. Montar Drive.
2. Verificar Python y GPU.
3. Instalar dependencias sin instalar `tensorflow-cpu==2.8.1`.
4. Preparar `openWakeWord` temporal.
5. Preparar Piper temporal desde el modelo persistente de Drive.
6. Aplicar parches idempotentes:
   - `torch.load(..., weights_only=False)`.
   - `openwakeword/data.py` usa `axis=1` solo en el lote NumPy.
   - entrenamiento ONNX-only, sin conversion TFLite.
7. Verificar recursos persistentes y descargar solo lo faltante desde URLs directas.
8. Crear o verificar `my_model_v2.yaml`.
9. Diagnosticar y normalizar los 5000 WAV generados a mono, 16 kHz, PCM16.
10. Generar features con log persistente:

```text
/content/drive/MyDrive/orion_wakeword_training/augment_clips.log
```

11. Verificar las cuatro features.
12. Entrenar y exportar:

```text
/content/drive/MyDrive/orion_wakeword_training/exports/orion_v2.onnx
```

## Recuperar Despues De Una Desconexion

Vuelve a ejecutar desde la primera celda. El notebook monta Drive, reconstruye los repositorios temporales en `/content`, reaplica parches y reutiliza lo que ya existe en Drive.

Si se interrumpio durante `--augment_clips`, la celda elimina solo las features `.npy` parciales o vacias y conserva todos los WAV. Luego vuelve a generar las features y guarda stdout/stderr completos en `augment_clips.log`.

Si las cuatro features ya existen y son validas, puedes continuar directamente con la verificacion de features y entrenamiento. Si ya existe:

```text
/content/drive/MyDrive/orion_wakeword_training/exports/orion_v2.onnx
```

puedes ejecutar la ultima celda para descargarlo.

## Que No Hace

- No entrena dentro de la `.venv` principal de ORION.
- No modifica `main.py`.
- No toca Whisper, Ollama, Policy Engine, ExecutionService ni el servicio wake word local.
- No usa AudioSet.
- No convierte a TFLite.
- No sube audio ni conecta APIs externas.
- No guarda `.onnx`, `.pt`, `.npy`, audios ni datasets en Git.
