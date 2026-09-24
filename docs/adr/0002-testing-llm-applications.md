# ADR 0002 - Como testar uma aplicação de LLM sem rede

- Status: aceito
- Data: 2026-09-23

## Contexto

Chamar a API real em testes é lento, custa dinheiro, depende de chave e produz saídas não
determinísticas. Ao mesmo tempo, testar só com mocks não prova que o request HTTP e o
streaming funcionam de verdade.

## Decisão: três níveis

| Nível | O que é | O que prova | Roda por padrão |
|---|---|---|---|
| Unitário | `ScriptedChatModel` (BaseChatModel falso que grava os prompts) | prompt enviado, histórico, janela, validação, tradução de erros, tags do LangSmith, CLI | sim |
| Wire | cliente **real** `langchain-openai` contra um servidor HTTP local que imita `/v1/chat/completions` (SSE inclusive) e também o endpoint do LangSmith | corpo do request (modelo, temperatura, mensagens), header `Authorization`, parsing de streaming, HTTP 401/429/500 -> erros de domínio, e que traces **saem do processo** só quando o tracing está ligado (subprocesso novo, com controle negativo) | sim |
| Live | uma chamada real (OpenAI, ou qualquer endpoint compatível via `OPENAI_BASE_URL`, como a Groq gratuita) com a pergunta do enunciado | integração de ponta a ponta | não (`-m live`, exige chave e custa); sem chave é *skipped* com o motivo |

Complementos: `filterwarnings = error`, cobertura mínima de 95% e nenhuma dependência de
variáveis do ambiente do desenvolvedor (fixture `autouse` limpa `OPENAI_*`/`LANGSMITH_*`).

## Consequências

* A suíte inteira roda offline em segundos e em CI sem segredos.
* O teste live é honesto sobre seu custo e nunca roda sem pedido explícito.
* Qualidade *semântica* das respostas (o modelo explica bem?) não é testável de forma
  determinística; o caminho recomendado é avaliação em datasets no LangSmith
  (ver ADR 0003).
