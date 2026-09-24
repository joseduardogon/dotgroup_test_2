# ADR 0003 - Observabilidade com LangSmith e caminho de avaliação

- Status: aceito
- Data: 2026-09-23

## Contexto

Aplicações de LLM falham de forma sutil (respostas ruins, custo, latência). Sem traces
não há como depurar nem comparar versões do prompt.

## Decisão

* **Tracing opt-in** por configuração: `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`
  (nomes oficiais do SDK). `configure_tracing` exporta as variáveis resolvidas (inclusive
  as vindas de `.env`, que o SDK não lê) antes da primeira chamada. Tracing ligado sem
  chave é **erro de configuração no startup**, não ausência silenciosa de traces.
* Cada execução recebe `run_name="python_assistant"`, tags (`prompt:<versão>`,
  `model:<nome>`) e metadados (`session_id`, `history_messages`, `prompt_version`), o que
  permite filtrar e comparar no LangSmith. Um teste garante essas marcas usando
  `collect_runs`, sem rede.
* **Prova de envio**: o SDK do LangSmith faz cache de leituras do ambiente (`lru_cache`),
  então exportar variáveis depois dos imports poderia falhar em silêncio. Um teste roda
  um interpretador novo (como a CLI) com `LANGSMITH_ENDPOINT` apontando para um servidor
  local e exige um `POST /runs...`; o mesmo cenário com tracing desligado (controle
  negativo) não pode enviar nada. Isso valida o mecanismo; o painel real foi conferido
  manualmente pelo autor (execução com as tags `prompt:1.0` e `model:...` e os metadados
  visíveis no projeto), sem teste automatizado contra o serviço.
* O prompt tem versão explícita (`PROMPT_VERSION`) e vive em código, revisável em PR.
* `pychat check` valida a configuração sem chamar serviço externo.

## Fora do escopo (próximo passo natural)

Avaliação automática: um dataset no LangSmith com `examples/questions.txt` e avaliadores
(LLM-as-judge para completude/correção; verificação de que há bloco de código
executável). Não foi incluído porque não pode ser verificado sem credenciais e sem
gastar chamadas reais.

## Consequências

Traces prontos para uso ao ligar uma variável; nenhum dado sai da máquina por padrão.
Atenção: traces contêm as perguntas dos usuários; em produção, tratar como dado pessoal.
