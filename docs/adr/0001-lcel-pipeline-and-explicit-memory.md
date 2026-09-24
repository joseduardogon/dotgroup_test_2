# ADR 0001 - Pipeline LCEL com memória explícita

- Status: aceito
- Data: 2026-09-23

## Contexto

O enunciado pede um chatbot que use o LangChain "para gerenciar o fluxo de conversação
e integrar com o LLM". O ecossistema oferece várias formas: `LLMChain`/`ConversationChain`
(legadas), `RunnableWithMessageHistory`, agentes/LangGraph e LCEL puro.

## Decisão

Um pipeline **LCEL** `ChatPromptTemplate | ChatOpenAI`,
com **memória explícita** mantida pela aplicação (`ConversationMemory`).

* `ChatPromptTemplate` com `MessagesPlaceholder("history")`: regras (system), histórico e
  pergunta ficam separados; chaves `{}` digitadas pelo usuário nunca são interpretadas
  como variáveis do template (há teste).
* Sem `StrOutputParser`: ele descartaria o `finish_reason`, que a execução real mostrou
  ser necessário para avisar quando uma resposta foi cortada pelo limite de tokens.
* O modelo é injetado como `BaseChatModel`: OpenAI em produção, `ScriptedChatModel` nos
  testes. Trocar de provedor é mudar uma fábrica.
* LCEL entrega `invoke`, `stream`, `batch` e tracing do LangSmith sem código extra.
* **Memória explícita** em vez de `RunnableWithMessageHistory`: o que é reenviado ao
  modelo fica visível, limitado (janela em mensagens, cortada em fronteira de troca
  completa) e testável; só trocas concluídas são gravadas, então uma falha ou Ctrl-C no
  meio da resposta não deixa contexto sujo. Também evita APIs em migração entre versões
  do LangChain.

## Alternativas descartadas

* `ConversationChain` e afins: legadas, sem streaming de primeira classe.
* Agentes/LangGraph: sem ferramentas nem fluxo condicional, seriam cerimônia. O caminho de
  evolução (RAG sobre a documentação do Python, ferramenta de execução de código em
  sandbox, checkpointer persistente) está aberto, pois o pipeline já é um `Runnable`.

## Consequências

Poucas linhas, comportamento previsível e cobertura de ~99% sem rede. Memória em processo:
para múltiplas instâncias seria necessário um repositório externo (Redis/SQL) atrás da
mesma interface.
