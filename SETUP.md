# Guia de Instalação — Case Otimização de Pilhas

Passo a passo para configurar o ambiente Python e rodar a aplicação.

## Pré-requisitos

- Python 3.10 ou superior instalado ([download](https://www.python.org/downloads/))
- Terminal (Prompt de Comando, PowerShell ou terminal do macOS/Linux)
- Dependências do projeto instaladas no ambiente virtual

## 1. Criar o ambiente virtual (venv)

No terminal, navegue até a pasta do projeto e execute:

```bash
python -m venv venv
```

Isso cria uma pasta `venv/` com uma instalação isolada do Python.

> **Por que usar venv?** Ambientes virtuais evitam conflitos entre bibliotecas de projetos diferentes. Cada projeto tem suas próprias dependências isoladas.

## 2. Ativar o ambiente virtual

### Windows (Prompt de Comando)

```bash
venv\Scripts\activate
```

### Windows (PowerShell)

```bash
venv\Scripts\Activate.ps1
```

> Se der erro de permissão no PowerShell, execute antes:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### macOS / Linux

```bash
source venv/bin/activate
```

Quando o venv estiver ativo, você verá `(venv)` no início da linha do terminal.

## 3. Instalar as dependências

Com o venv ativado, execute:

```bash
pip install -r requirements.txt
```

## 4. Rodar a aplicação

```bash
streamlit run model.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`.

## 5. Desativar o ambiente virtual

Quando terminar, desative o venv com:

```bash
deactivate
```

## Resumo rápido dos comandos

| Etapa | Comando |
|-------|---------|
| Criar venv | `python -m venv venv` |
| Ativar (Windows CMD) | `venv\Scripts\activate` |
| Ativar (macOS/Linux) | `source venv/bin/activate` |
| Instalar dependências | `pip install -r requirements.txt` |
| Rodar aplicação | `streamlit run model.py` |
| Desativar venv | `deactivate` |
