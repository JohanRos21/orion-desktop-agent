# O.R.I.O.N. Desktop Agent

**Operational Reasoning, Interaction and Orchestration Network**

Agente inteligente de escritorio para Windows desarrollado progresivamente desde automatización básica hasta lenguaje natural, voz, memoria y control seguro del sistema.

## Estado actual

**V0.1 — Motor inicial de comandos**

Actualmente ORION puede:

- recibir instrucciones escritas;
- normalizar texto;
- reconocer comandos básicos;
- validar aplicaciones permitidas;
- abrir aplicaciones de Windows;
- rechazar acciones desconocidas;
- devolver resultados estructurados.

## Ejecutar

```powershell
.\.venv\Scripts\Activate.ps1
python -m orion.main