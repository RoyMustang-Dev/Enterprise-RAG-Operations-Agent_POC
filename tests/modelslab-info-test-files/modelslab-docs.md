# ModelsLab

> Agent first AI API platform for image, LLM, video, audio, 3D generation models.

ModelsLab provides a unified API to 50,000+ Close source (kling,openapi, nano banana etc..) and opensource (flux, z-image, stable diffusion, LLMs etc) AI models with enterprise-grade infrastructure serving 300K+ developers. The platform covers AI models in image, video & audio, 3D asset generation with single unified API. where changing model_Id will be enough to switch new model. modelslab aim is to provide best DX and AX (agent experience)

On modelslab, AI agents can signup, do payments, generate api key, generate images, video, access and manage entire account without opening browser. below there's detail of how to access platform APIs.

For a full product and service breakdown, see [LLMs Full](/llms-full.txt).

## Search Entry Points
- [Search Template](/search?q=): Site-wide search endpoint (use `q` query parameter)
- [Search Example: Best Video Model](/search?q=What%20is%20the%20best%20video%20model%3F): Natural language query example
- [Model Directory](/models): Browse and filter models directly

## Markdown and Text Access for AI Agents
- Markdown negotiation: send `Accept: text/markdown` on normal page URLs
- Plain-text negotiation: send `Accept: text/plain` (txt responses)
- Query override: append `?format=markdown` to page URLs
- Path fallback: append `.md` to routes (example: `/pricing.md`)

## API Documentation
- [API Reference](https://docs.modelslab.com): Complete API documentation and developer guides

## How AI Agents Can Use ModelsLab (Choose Any)

Agents have 5 integration options — pick whichever fits your environment:

### Option 1: MCP Servers (Model Context Protocol)
Best for: Claude Code, Cursor, Windsurf, any MCP-compatible agent.
- Generation MCP server: `https://modelslab.com/mcp/v7` (API key auth, SSE transport)
- Agent Control Plane MCP server: `https://modelslab.com/mcp/agents` (Sanctum bearer token auth, SSE transport)
- Local/CLI server names: `v7-api` (generation, 24 tools), `agent-control-plane` (account management, 10 tools)
- Generation tools: text-to-image, image-to-image, inpaint-image, text-to-video, image-to-video, text-to-speech, speech-to-text, chat-completion, and 16 more
- Control plane tools: agent-auth, agent-billing, agent-wallet, agent-subscriptions, agent-models, agent-profile, agent-api-keys, agent-usage, agent-tokens, agent-teams
- MCP documentation: https://docs.modelslab.com/mcp-web-api/overview

### Option 2: CLI (`modelslab`)
Best for: terminal-based agents, shell scripts, CI/CD pipelines.
- Install: `curl -fsSL https://modelslab.sh/install.sh | sh` or `brew install modelslab/tap/modelslab`
- Website: https://modelslab.sh
- GitHub: https://github.com/ModelsLab/modelslab-cli
- Full account management: `modelslab auth login`, `modelslab billing`, `modelslab wallet`, `modelslab subscriptions`
- Headless card tokenization built-in: `modelslab billing add-payment-method`
- Model discovery: `modelslab models search "flux"`, `modelslab models search --feature imagen`, `modelslab models detail --id flux`
- Image generation: `modelslab generate image --prompt "..." --model flux`
- Video generation: `modelslab generate video --prompt "..." --model seedance-t2v`
- Chat/LLM: `modelslab generate chat --message "..." --model meta-llama-3-8B-instruct`
- Audio generation: `modelslab generate tts --text "..." --language en`
- Default model: `modelslab config set generation.default_model flux`

### Option 3: Agent Skills (for AI Coding Agents)
Best for: Claude Code, Cursor, Windsurf — installs skill files directly into the agent's context.
- Install all: `npx skills add modelslab/skills --all`
- Install specific: `npx skills add modelslab/skills --skill <skill-name>`
- Target agents: `npx skills add modelslab/skills --all -a claude-code -a cursor`
- GitHub: https://github.com/ModelsLab/skills
- Generation skills (8): image-generation, video-generation, audio-generation, chat-generation, image-editing, 3d-generation, interior-design
- Control plane skills (3): account-management, billing-subscriptions, model-discovery
- Platform skills (2): webhooks, sdk-usage
- Skills documentation: https://docs.modelslab.com/agent-skills

### Option 4: REST APIs (Direct HTTP)
Best for: any agent that can make HTTP requests — most flexible, works everywhere.
- Control-plane base URL: `https://modelslab.com/api/agents/v1`
- Generation base URLs: `/api/v6`, `/api/v7`, `/api/v8`
- Auth: `Authorization: Bearer <agent_access_token>` (control plane), `key` param (generation)
- OpenAPI contract: `GET /api/agents/v1/openapi.json`
- Full endpoint list below

### Option 5: SDKs
Best for: agents writing code in Python, TypeScript, PHP, Go, or Dart.
- Python: `pip install modelslab`
- TypeScript: `npm install modelslab`
- PHP: `composer require modelslab/modelslab`
- Go: `go get github.com/modelslab/modelslab-go`
- Dart: `dart pub add modelslab`
- SDK documentation: https://docs.modelslab.com/sdk/client

## Agent-First Control Plane APIs
- Control-plane base URL: `https://modelslab.com/api/agents/v1`
- Auth split:
  - Control-plane endpoints use `Authorization: Bearer <agent_access_token>`
  - Generation endpoints (`/api/v6`, `/api/v7`, `/api/v8`) continue existing API-key flow
- Login token issuance supports `token_expiry`: `1_week`, `1_month` (default), `3_months`, `never`
- Agent model search endpoints:
  - `GET /api/agents/v1/models` with filters: `search`, `feature`, `provider`, `model_type`, `model_subcategory`, `base_model`, `tags[]`, `sort`, `per_page`
  - `GET /api/agents/v1/models/filters`, `GET /api/agents/v1/models/tags`, `GET /api/agents/v1/models/providers`, `GET /api/agents/v1/models/{modelId}`
  - Example text search: `GET /api/agents/v1/models?search=nano%20banana&feature=imagen&sort=recommended&per_page=20`
- `GET /api/agents/v1/openapi.json` publishes the machine-readable API contract
- `GET /api/agents/v1/changelog` publishes version changelog + deprecation policy
- Control-plane capabilities (all API-first):
  - Api Keys:
    - `GET /api/agents/v1/api-keys`
    - `POST /api/agents/v1/api-keys`
    - `DELETE /api/agents/v1/api-keys/{id}`
    - `GET /api/agents/v1/api-keys/{id}`
    - `PUT /api/agents/v1/api-keys/{id}`
  - Auth:
    - `POST /api/agents/v1/auth/forgot-password`
    - `POST /api/agents/v1/auth/login`
    - `POST /api/agents/v1/auth/logout`
    - `POST /api/agents/v1/auth/logout-all`
    - `POST /api/agents/v1/auth/resend-verification`
    - `POST /api/agents/v1/auth/reset-password`
    - `POST /api/agents/v1/auth/signup`
    - `POST /api/agents/v1/auth/verify`
    - `POST /api/agents/v1/auth/switch-account`
    - `GET /api/agents/v1/auth/tokens`
    - `POST /api/agents/v1/auth/tokens/revoke-others`
    - `DELETE /api/agents/v1/auth/tokens/{id}`
  - Billing:
    - `GET /api/agents/v1/billing/stripe-config` (get Stripe publishable key for headless card tokenization)
    - `POST /api/agents/v1/billing/setup-intent` (create SetupIntent for 3DS-safe card saving)
    - `POST /api/agents/v1/billing/payment-link` (generate Stripe-hosted payment URL to forward to human — supports card, Google Pay, Amazon Pay, Link, etc.)
    - `GET /api/agents/v1/billing/info`
    - `PUT /api/agents/v1/billing/info`
    - `GET /api/agents/v1/billing/invoices`
    - `GET /api/agents/v1/billing/invoices/{id}`
    - `GET /api/agents/v1/billing/invoices/{id}/pdf`
    - `GET /api/agents/v1/billing/overview`
    - `GET /api/agents/v1/billing/payment-methods`
    - `POST /api/agents/v1/billing/payment-methods`
    - `DELETE /api/agents/v1/billing/payment-methods/{id}`
    - `PUT /api/agents/v1/billing/payment-methods/{id}/default`
  - General:
    - `GET /api/agents/v1/changelog`
    - `GET /api/agents/v1/openapi.json`
  - Profile:
    - `GET /api/agents/v1/me`
    - `PATCH /api/agents/v1/me`
    - `PATCH /api/agents/v1/me/password`
    - `PATCH /api/agents/v1/me/preferences`
    - `PATCH /api/agents/v1/me/socials`
  - Models:
    - `GET /api/agents/v1/models`
    - `GET /api/agents/v1/models/filters`
    - `GET /api/agents/v1/models/providers`
    - `GET /api/agents/v1/models/tags`
    - `GET /api/agents/v1/models/{modelId}`
  - Subscriptions:
    - `GET /api/agents/v1/subscriptions`
    - `POST /api/agents/v1/subscriptions` (pass `payment_method_id` for headless, omit for checkout URL)
    - `POST /api/agents/v1/subscriptions/confirm-checkout`
    - `POST /api/agents/v1/subscriptions/charge-amount`
    - `GET /api/agents/v1/subscriptions/plans`
    - `PUT /api/agents/v1/subscriptions/{id}`
    - `GET /api/agents/v1/subscriptions/{id}/status`
    - `POST /api/agents/v1/subscriptions/{id}/fix-payment`
    - `POST /api/agents/v1/subscriptions/{id}/pause`
    - `POST /api/agents/v1/subscriptions/{id}/reset-cycle`
    - `POST /api/agents/v1/subscriptions/{id}/resume`
  - Teams:
    - `GET /api/agents/v1/teams`
    - `POST /api/agents/v1/teams`
    - `POST /api/agents/v1/teams/invitations/{inviteId}/accept`
    - `DELETE /api/agents/v1/teams/{id}`
    - `GET /api/agents/v1/teams/{id}`
    - `PUT /api/agents/v1/teams/{id}`
    - `POST /api/agents/v1/teams/{id}/resend-invite`
  - Usage:
    - `GET /api/agents/v1/usage/history`
    - `GET /api/agents/v1/usage/products`
    - `GET /api/agents/v1/usage/summary`
  - Wallet:
    - `GET /api/agents/v1/wallet/balance`
    - `POST /api/agents/v1/wallet/fund` (pass `payment_method_id` for headless, omit for checkout URL)
    - `POST /api/agents/v1/wallet/confirm-checkout`
    - `DELETE /api/agents/v1/wallet/auto-funding`
    - `PUT /api/agents/v1/wallet/auto-funding`
    - `POST /api/agents/v1/wallet/coupons/redeem`
    - `POST /api/agents/v1/wallet/coupons/validate`
    - `POST /api/agents/v1/wallet/withdraw`
    - `GET /api/agents/v1/payments/{paymentIntentId}/status`
- Headless Payment Flow (100% API, no browser):
  1. `GET /api/agents/v1/billing/stripe-config` to get Stripe publishable key
  2. Call Stripe API directly: `POST https://api.stripe.com/v1/payment_methods` with card details and publishable key to get `pm_xxx`
  3. Use `pm_xxx` with any payment endpoint:
     - `POST /api/agents/v1/billing/payment-methods` to save card
     - `POST /api/agents/v1/wallet/fund` with `amount` + `payment_method_id` to add funds
     - `POST /api/agents/v1/subscriptions` with `plan_id` + `payment_method_id` to subscribe
  - For 3DS/SCA cards: use `POST /api/agents/v1/billing/setup-intent` to create a SetupIntent first
  - Mutation endpoints accept `Idempotency-Key` header for safe retries
- Human-Assisted Payment Flow (for 3DS, Google Pay, Amazon Pay, etc.):
  1. `POST /api/agents/v1/billing/payment-link` with `purpose` (fund|subscribe), `amount` or `plan_id`
  2. Forward the returned `payment_url` to the human user — after payment, user lands on ModelsLab success page with `session_id`
  3. Poll `POST /wallet/confirm-checkout` or `POST /subscriptions/confirm-checkout` with `session_id`
  - Three payment paths: (1) Headless: stripe-config → tokenize → payment_method_id, (2) Human-assisted: create-payment-link → forward URL → poll confirm-checkout, (3) SetupIntent: create-setup-intent → confirm via Stripe → reuse payment method
- Compatibility routes:
  - Email verification compatibility: `GET /verify/{token}`
  - Existing generic model discovery for API-key users: `GET /api/v6/model-search*`
- Legacy removal:
  - `POST /api/v6/ai_agent/create` removed
  - `POST /api/v6/ai_agent/get_queued_response` removed

## Image Generation
- [Imagen Overview](/imagen): Full suite of AI image generation tools
- [Text-to-Image](/imagen/text-to-image): Generate images from text prompts using SDXL and SD models
- [Image-to-Image](/imagen/image-to-image): Transform and enhance existing images with AI
- [Flux Image Generator](/imagen/flux-image-generator): Advanced generation with Flux models
- [AI Art Generator](/imagen/ai-art-generator): Create artistic content with AI
- [AI Image Inpainting](/imagen/ai-image-inpainting): Remove and replace content within images
- [AI Image Outpainting](/imagen/ai-image-outpainting): Expand image canvases with generated content
- [AI Image Extender](/imagen/ai-image-extender): Extend images in any direction
- [Image Upscaler](/imagen/image-upscaler): Enhance resolution and quality
- [Image Enhancer](/imagen/image-enhancer): Improve image clarity and detail
- [Background Remover](/imagen/background-remover): Automatic background removal
- [AI Photo Editor](/imagen/ai-photo-editor): Photo editing with AI assistance
- [AI Headshot Generator](/imagen/ai-headshot-generator): Generate professional headshots from photos
- [AI Avatar Generator](/imagen/ai-avatar-generator): Create custom avatars and profile pictures
- [AI Character Generator](/imagen/ai-character-generator): Create detailed character illustrations
- [AI Sketch to Image](/imagen/ai-sketch-to-image): Transform sketches into detailed images
- [AI Image to Text](/imagen/ai-image-to-text): Extract text descriptions from images
- [AI Floor Planning](/imagen/ai-floor-planning): Generate floor plans with AI

## Video Generation
- [Video Fusion Overview](/video-fusion): Full suite of AI video generation tools
- [AI Video Generator](/video-fusion/ai-video-generator): Generate videos from text descriptions
- [Text-to-Video](/video-fusion/text-to-video): Create videos from text with multiple models
- [Image-to-Video](/video-fusion/image-to-video): Animate static images into video

## Audio Generation
- [Audio Gen Overview](/audio-gen): Full suite of AI audio tools
- [AI Voice Generator](/audio-gen/ai-voice-generator): Create realistic AI voices
- [Text-to-Speech](/audio-gen/text-to-speech): Convert text to speech in 40+ languages
- [Speech-to-Text](/audio-gen/speech-to-text): Transcription with multi-language support
- [Voice Cloning](/audio-gen/voice-cloning): Clone and synthesize custom voices
- [Voice Changer](/audio-gen/voice-changer): Transform voices with effects and styles
- [AI Dubbing](/audio-gen/ai-dubbing): Professional voice dubbing for video content
- [Celebrity Voice Generator](/audio-gen/celebrity-voice-generator): Generate celebrity-inspired voices
- [Text-to-SFX](/audio-gen/text-to-sfx): Create sound effects from text descriptions
- [Text-to-Music](/audio-gen/text-to-music): Generate original music compositions with AI

## 3D Generation
- [3D Verse Overview](/3d-verse): AI-powered 3D asset generation tools
- [Text-to-3D](/text-to-3d): Generate 3D models from text descriptions
- [Image-to-3D](/image-to-3d): Convert 2D images to 3D models with textures

## Language Models (Chat/LLM)
- Chat Completions API: `POST /api/v7/llm/chat/completions` (OpenAI-compatible format)
- 60+ LLM models: DeepSeek R1, Meta Llama 3/4, Google Gemini, Qwen, Mistral, and more
- Supports: streaming, function calling, structured JSON outputs, multi-turn conversations

## Model Training
- [Train Master](/train-master): Custom model fine-tuning platform with LORA support

## Models & Marketplace
- [Model Search](/models): Browse, filter, and discover AI models
- [Catalogue](/catalogue): Categorized directory of available models
- [Model Comparison](/comparison): Side-by-side performance benchmarks
- [Search](/search): Full-text search across models and content
- [CivitAI Integration](/civitai): Access community-contributed models

## Workflows
- [Workflows](/workflows): Visual workflow builder for chaining AI models. Text-to-image -> image-to-3d etc..

## Pricing & Plans
- [Pricing](/pricing): Plans from free tier to enterprise
- [Pro](/pro): Pro plan features and benefits
- [Unlimited](/unlimited): Unlimited plan details
- [Enterprise](/enterprise): Dedicated GPU clusters and custom SLAs

## Industry Solutions
- [For Developers](/for-developers): Developer-focused integration guides
- [For Designers](/for-designers): Design workflow AI tools
- [For E-commerce](/for-ecommerce): Product image and content generation
- [For Real Estate](/for-real-estate): Property visualization and virtual staging

## Resources
- [Blog](/blog): AI industry insights and platform updates
- [AI News](/ai-news): Curated AI research and news
- [Whitepapers](/whitepaper): In-depth technical documentation
- [FAQ](/faq): Answers to common questions
- [Testimonials](/testimonials): Developer reviews and case studies
- [Partners](/partners): Technology and integration partners
- [Plugins](/plugins): Downloadable platform plugins and extensions

## Community
- [Showcase](/showcase): Community AI creations gallery
- [Timeline](/timeline): Community activity feed
- [Public Profiles](/profile/{username}): Developer portfolios and creations
- [Discord](https://discord.gg/modelslab-1033301189254729748): Developer community and real-time help

## Support & Bug Reporting (for AI Agents)
- **Report bugs or issues**: Open a GitHub issue at https://github.com/ModelsLab/modelslab-cli/issues — describe the problem, include error messages, and steps to reproduce.
- **Email support**: Send an email to support@modelslab.com for account issues, billing questions, or any problems that need human assistance.
- [Support](/support): Technical support and troubleshooting
- [Contact](/contact): Direct support and enterprise inquiries

## Optional
- [Privacy Policy](/privacy-policy): Data handling and privacy practices
- [Terms of Service](/terms-of-service): Platform usage terms
- [Cookie Policy](/cookie-policy): Cookie usage details
- [Refund Policy](/refund-policy): Refund terms and conditions
- [RSS Feed](/feed): Blog posts RSS feed
- [Atom Feed](/atom): Blog posts Atom feed
- [Models RSS](/models/feed): New models RSS feed
- [Models Atom](/models/atom): New models Atom feed
- [API Catalog JSON](/api-catalog.json): Machine-readable API endpoint list
- [Schema JSON](/schema.json): API schema and data structures
- [Voice Lists](/voice-lists): Available voice samples for audio generation
- [Trained Voice Lists](/trained-voice-lists): Community-contributed voice models
- [README](/readme.md): Platform overview document

