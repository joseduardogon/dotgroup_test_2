# ADR 0004 - Segurança e controle de custo

- Status: aceito
- Data: 2026-09-23

## Decisões

| Risco | Controle |
|---|---|
| Vazamento de chaves | `SecretStr` (fora de `repr`/logs), nunca no código nem na imagem Docker (`--env-file`), `.env` no `.gitignore`, `pychat check` mostra "configured (hidden)"; testes garantem |
| Prompt injection | System prompt trata todo texto do usuário como pergunta, nunca como instrução, e recusa assuntos fora de Python; **é mitigação, não garantia**: o assistente não tem ferramentas nem acesso a dados sensíveis, então o impacto é limitado a respostas |
| Custo descontrolado | limite de tamanho da pergunta, `max_tokens`, janela de histórico, retries limitados, entrada vazia rejeitada antes de chamar a API |
| Falhas do provedor | retry com backoff no cliente; erros traduzidos (401, 429, rede, modelo inexistente) com dica de correção; REPL continua após falha |
| Estado inconsistente | só trocas completas entram no histórico |
| Container | usuário não-root, filesystem read-only, `no-new-privileges` |

## Limites conhecidos

Sem rate limiting por usuário nem autenticação (é uma CLI local). Se virar serviço HTTP,
adicionar ambos e mover a memória para armazenamento compartilhado.
