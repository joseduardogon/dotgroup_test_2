# PyMentor - Chatbot de Python com LangChain, OpenAI e LangSmith

> **Autor:** José Eduardo Gontijo de Carvalho - GitHub [@joseduardogon](https://github.com/joseduardogon)
>
> **Contexto:** Questão 2 do teste técnico da **Dot Group** para a vaga de
> *Senior Developer* com foco em **IA e Backend**.

---

## English summary

A terminal chatbot that answers Python programming questions with an OpenAI chat model,
orchestrated by LangChain (LCEL: `prompt | model | parser`), with streaming, bounded
conversation memory, friendly error handling and optional LangSmith tracing. The model is
injected, so the whole suite (unit, wire-level and CLI tests) runs offline without an API
key; a single opt-in test exercises the real API. Try it: `pychat ask "Como criar uma lista
em Python?"`. Design decisions are in `docs/adr/`, sample Q&A in `docs/examples.md`.

---

## 1. A questão

**Questão 2 - Implementação de Chatbot com IA Generativa (Langchain, Langsmith, LLMs)**

> Você precisa desenvolver um chatbot que utilize um modelo de linguagem (LLM) como o GPT-4
> da OpenAI para responder perguntas dos usuários sobre programação em Python. O chatbot
> deve:
>
> 1. Receber perguntas dos usuários via input de texto.
> 2. Utilizar o Langchain para gerenciar o fluxo de conversação e integrar com o LLM.
> 3. Responder às perguntas utilizando o modelo da OpenAI.
>
> Implemente um exemplo simples onde o usuário possa perguntar algo como "Como criar uma
> lista em Python?" e o chatbot responda com uma explicação detalhada.
>
> Dicas: utilize o Langchain para facilitar a integração e gerenciamento das respostas do
> LLM; certifique-se de configurar corretamente a API da OpenAI; forneça exemplos de
> perguntas e respostas para demonstrar o funcionamento do chatbot.

### Requisitos x entrega

| Requisito | Entrega |
|---|---|
| Receber perguntas via texto | `pychat chat` (conversa interativa) e `pychat ask "..."` (pergunta única) |
| LangChain gerencia o fluxo e integra com o LLM | pipeline LCEL `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` + memória de conversa |
| Responder com o modelo da OpenAI | `langchain-openai` (`gpt-4o` por padrão; qualquer modelo via `PYCHAT_MODEL`, ex.: `gpt-4`) |
| LangSmith (título da questão) | tracing opt-in com tags e metadados por execução |
| Configurar corretamente a API da OpenAI | `OPENAI_API_KEY` via ambiente/`.env`, validada por `pychat check` |
| Exemplos de perguntas e respostas | [docs/examples.md](docs/examples.md) e [examples/questions.txt](examples/questions.txt) |

Extras: streaming token a token com renderização Markdown, memória por sessão, erros
traduzidos com dicas, `--raw` para pipes, Docker, CI, ADRs, testes sem rede.

---

## 2. Como executar

### Pré-requisitos

Python 3.12+, Poetry >= 2.0 e uma chave da OpenAI (`OPENAI_API_KEY`).

```bash
poetry install
cp .env.example .env        # edite e coloque sua chave (o arquivo é ignorado pelo git)
poetry run pychat check     # valida a configuração sem chamar nenhuma API
poetry run pychat chat      # conversa interativa
poetry run pychat ask "Como criar uma lista em Python?"
```

Comandos do `chat`: `/help`, `/reset` (esquece a conversa), `/exit` (ou Ctrl-D).
`pychat ask --raw "..."` imprime texto puro, ideal para pipes.

### Docker

```bash
docker build -t dotgroup-test-2 .
docker run --rm -it --env-file .env dotgroup-test-2            # abre o chat
docker run --rm --env-file .env dotgroup-test-2 ask "Como criar uma lista em Python?"
```

A chave nunca entra na imagem: é passada em tempo de execução. Também há
`docker compose run --rm chat`.

### Configuração

| Variável | Padrão | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | - | credencial da OpenAI (obrigatória) |
| `OPENAI_BASE_URL` | oficial | endpoint alternativo (proxy, gateway, servidor compatível) |
| `PYCHAT_MODEL` | `gpt-4o` | modelo de chat (ex.: `gpt-4`, `gpt-4o-mini`) |
| `PYCHAT_TEMPERATURE` | `0.2` | baixa para respostas de código mais determinísticas |
| `PYCHAT_MAX_TOKENS` | `1024` | limite de tokens por resposta |
| `PYCHAT_MAX_HISTORY_MESSAGES` | `20` | janela de contexto reenviada ao modelo |
| `PYCHAT_MAX_QUESTION_CHARS` | `4000` | tamanho máximo da pergunta |
| `PYCHAT_TIMEOUT_SECONDS`, `PYCHAT_MAX_RETRIES` | `60`, `2` | rede e retentativas |
| `LANGSMITH_TRACING` | `false` | envia traces ao LangSmith |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | - | credencial, projeto e endpoint do LangSmith |

Ligar o tracing sem `LANGSMITH_API_KEY` é um erro de configuração já na inicialização.

### Qualidade e testes

```bash
make check                                  # ruff + mypy --strict + pytest com cobertura (>= 95%)
poetry run pytest --no-cov tests/unit       # apenas unitários
poetry run pytest --no-cov -m integration   # cliente real da OpenAI contra servidor HTTP local
poetry run pytest --no-cov -m live          # UMA chamada real e paga à OpenAI (exige chave)
```

---

## 3. Como foi implementado

### 3.1 Visão geral

```
pychat (Typer/Rich) ──► PythonAssistant ──► prompt | ChatOpenAI | StrOutputParser ──► OpenAI
   cli/                   chat/assistant.py        (LCEL, llm/factory.py)               ▲
                              │  ▲                                                      │
                     ConversationMemory  └── traces (LangSmith) ────────────────────────┘
                          chat/memory.py        core/tracing.py
```

Dependências apontam para dentro: `cli -> chat -> llm/core`. Apenas o pacote `cli` lê ou
imprime no terminal; o restante é testável sem TTY e sem rede.

```
src/python_chatbot/
  core/    config.py (Settings), exceptions.py, tracing.py
  llm/     factory.py           único lugar que conhece a OpenAI
  chat/    prompts.py, memory.py, assistant.py, factory.py (composition root)
  cli/     app.py (comandos), repl.py (loop), render.py (Markdown/erros)
tests/     unit/ · integration/ (wire + live) · fakes.py
docs/adr/  decisões de arquitetura
```

### 3.2 O pipeline LangChain (`chat/assistant.py`, `chat/prompts.py`)

`PythonAssistant` monta `build_prompt() | model | StrOutputParser()`
([ADR 0001](docs/adr/0001-lcel-pipeline-and-explicit-memory.md)):

1. **Prompt** (`ChatPromptTemplate`): system prompt versionado (persona de instrutor
   Python: resposta direta + exemplo executável em bloco `python` + explicação e
   armadilhas; mesma língua do usuário; Python 3.12+; recusa fora do escopo; nunca inventar
   APIs; trata a entrada do usuário como pergunta, não como instrução), um
   `MessagesPlaceholder` para o histórico e a pergunta.
2. **Modelo**: `ChatOpenAI` construído em `llm/factory.py` a partir das configurações e
   injetado como `BaseChatModel`.
3. **Parser**: `StrOutputParser` devolve texto puro.

`ask()` faz uma chamada única; `stream()` devolve fragmentos à medida que chegam e
grava a troca na memória **somente se terminar com sucesso**.

### 3.3 Memória de conversa (`chat/memory.py`)

Estado explícito, por sessão, thread-safe e limitado: guarda as últimas N mensagens sempre
cortadas em fronteira de troca (nunca começa por uma resposta órfã). Perguntas de
acompanhamento ("e como adiciono um item nela?") funcionam porque o histórico é reenviado.
`/reset` ou `reset()` limpa a sessão; sessões diferentes não se misturam.

### 3.4 Erros e robustez (`core/exceptions.py`, `chat/assistant.py`)

Falhas do SDK da OpenAI viram uma hierarquia de domínio, cada uma com dica de correção e
código de saída próprio: chave inválida (`LLMAuthenticationError`), limite/cota
(`LLMRateLimitError`), rede/5xx (`LLMUnavailableError`), modelo inexistente/requisição
recusada (`LLMRequestError`). Perguntas vazias ou longas demais são barradas **antes** de
gastar uma chamada. O REPL continua após um erro. Respostas vazias (ex.: filtro de
conteúdo) são reportadas e não gravadas.

### 3.5 Observabilidade (`core/tracing.py`, [ADR 0003](docs/adr/0003-observability-and-evaluation.md))

Com `LANGSMITH_TRACING=true`, cada execução vira um trace com `run_name`, tags
(`prompt:1.0`, `model:<nome>`) e metadados (`session_id`, `history_messages`). As variáveis
vindas do `.env` são exportadas antes da primeira chamada, já que o SDK só lê o ambiente.

### 3.6 Segurança e custo ([ADR 0004](docs/adr/0004-security-and-cost-controls.md))

`SecretStr` para credenciais (não aparecem em `repr`/logs), chave fora do código e da
imagem, limites de tamanho/tokens/histórico/retentativas, container não-root e read-only.

### 3.7 Estratégia de testes ([ADR 0002](docs/adr/0002-testing-llm-applications.md))

- **Unitários**: `ScriptedChatModel` (modelo falso que grava os prompts) verifica o
  prompt enviado, o histórico, a janela, a validação, a tradução de erros, as tags do
  LangSmith e a CLI de ponta a ponta.
- **Wire**: o cliente **real** `langchain-openai` fala HTTP com um servidor local que imita
  `/v1/chat/completions` (incluindo streaming SSE), validando o corpo do request, o header
  `Authorization`, o parsing do stream e os erros HTTP 401/429/500.
- **Live** (opt-in): uma chamada real com a pergunta do enunciado.

Tudo, exceto o live, roda offline e sem chave. `filterwarnings = error`, cobertura mínima
de 95% e mypy estrito.

### 3.8 Limitações conhecidas

- **A chamada real à OpenAI não foi executada no desenvolvimento** (sem chave disponível):
  o caminho de rede é validado pelo servidor local compatível, e o teste `-m live` fica
  pronto para você rodar. As respostas em `docs/examples.md` são ilustrativas.
- Memória em processo (uma instância); avaliação automática no LangSmith é o próximo
  passo natural ([ADR 0003](docs/adr/0003-observability-and-evaluation.md)).
- Modelos de raciocínio (o-series/GPT-5) não aceitam qualquer `temperature`; ajuste
  `PYCHAT_TEMPERATURE` conforme o modelo escolhido.

## 4. Licença

MIT - veja [LICENSE](LICENSE).
