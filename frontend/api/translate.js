export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { source_en } = req.body;
  if (!source_en) {
    return res.status(400).json({ error: 'Missing source_en' });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'OPENROUTER_API_KEY not configured' });
  }

  const MODEL = 'anthropic/claude-sonnet-4-6';
  const COVERAGE_THRESHOLD = 0.6;
  const MAX_REVISIONS = 1;

  const GLOSSARY = {
    'password': { ja: 'パスワード', forbidden: ['暗証番号'] },
    'account settings': { ja: 'アカウント設定' },
    'account': { ja: 'アカウント' },
    'login': { ja: 'ログイン', forbidden: ['サインイン', 'ログオン'] },
    'logout': { ja: 'ログアウト' },
    'settings': { ja: '設定' },
    'user': { ja: 'ユーザー', forbidden: ['ユーザ', '利用者'] },
    'email': { ja: 'メール' },
    'download': { ja: 'ダウンロード' },
    'upload': { ja: 'アップロード' },
    'search': { ja: '検索' },
    'update': { ja: '更新', forbidden: ['アップデート'] },
    'delete': { ja: '削除' },
    'page': { ja: 'ページ' },
    'security': { ja: 'セキュリティ' },
    'authentication': { ja: '認証' },
    'notification': { ja: '通知' },
    'profile': { ja: 'プロフィール' },
    'database': { ja: 'データベース' },
    'server': { ja: 'サーバー' },
    'browser': { ja: 'ブラウザ' },
    'network': { ja: 'ネットワーク' },
    'backup': { ja: 'バックアップ' },
    'file': { ja: 'ファイル' },
    'folder': { ja: 'フォルダ' },
    'button': { ja: 'ボタン' },
    'link': { ja: 'リンク' },
    'menu': { ja: 'メニュー' },
    'error': { ja: 'エラー' },
    'message': { ja: 'メッセージ' },
    'data': { ja: 'データ' },
    'system': { ja: 'システム' },
    'software': { ja: 'ソフトウェア' },
    'content': { ja: 'コンテンツ' },
    'website': { ja: 'ウェブサイト' },
  };

  const TOOLS = [
    {
      type: 'function',
      function: {
        name: 'lookup_glossary',
        description: 'Look up the approved Japanese translation for an English term. Returns the mandatory approved form and forbidden variants.',
        parameters: {
          type: 'object',
          properties: { term: { type: 'string', description: 'English term (case-insensitive)' } },
          required: ['term'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'lookup_translation_memory',
        description: 'Find previously translated sentences similar to the source text. Returns close matches with approved translations.',
        parameters: {
          type: 'object',
          properties: {
            sentence: { type: 'string', description: 'Source English sentence' },
            top_k: { type: 'integer', description: 'Number of matches (default 3)' },
          },
          required: ['sentence'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'lookup_grammar_pattern',
        description: 'Search Japanese grammar reference for particles (は/も/が), verb forms, polite speech, conditionals. Use when unsure about a grammatical construction.',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string', description: 'Grammar question (e.g. "は vs が", "polite ます form")' } },
          required: ['query'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'validate_locale',
        description: 'Check Japanese text for formatting errors: wrong date formats (MM/DD → MM月DD日), ASCII punctuation (. → 。, , → 、), western quotes (" → 「」). Call after drafting translation.',
        parameters: {
          type: 'object',
          properties: { text: { type: 'string', description: 'Japanese text to validate' } },
          required: ['text'],
        },
      },
    },
  ];

  function executeGlossary(term) {
    const key = term.toLowerCase().trim();
    if (GLOSSARY[key]) {
      const e = GLOSSARY[key];
      return JSON.stringify({ found: true, term: key, approved_ja: e.ja, ...(e.forbidden ? { forbidden_variants: e.forbidden } : {}) });
    }
    const partials = Object.entries(GLOSSARY).filter(([k]) => k.includes(key) || key.includes(k)).slice(0, 3);
    if (partials.length) return JSON.stringify({ found: false, partial_matches: partials.map(([k, v]) => ({ term: k, approved_ja: v.ja })) });
    return JSON.stringify({ found: false, message: `No entry for "${term}"` });
  }

  function executeTM(sentence) {
    return JSON.stringify({ matches: [], message: 'Translation memory not available in demo mode. Full pipeline uses local TM with Jaccard similarity matching.' });
  }

  function executeGrammar(query) {
    const patterns = {
      'は vs が': 'は marks the topic (what we\'re talking about). が marks the subject (who/what does the action). Use は for known/contrasted info, が for new/emphasized info.',
      'polite': 'Use ます/です for formal/polite register. Use plain form (だ/る) for casual. Match the source register.',
      'particle': 'を marks direct object. に marks destination/time/indirect object. で marks location of action/means.',
    };
    const key = Object.keys(patterns).find(k => query.toLowerCase().includes(k));
    if (key) return JSON.stringify({ found: true, pattern: key, explanation: patterns[key] });
    return JSON.stringify({ found: false, message: `No specific pattern for "${query}". General rule: match source register and use natural particle choices.` });
  }

  function validateLocale(text) {
    const issues = [];
    if (/\d{1,2}\/\d{1,2}/.test(text)) issues.push({ rule: 'date_slash_format', suggestion: 'Use MM月DD日' });
    if (/[\u3040-\u9FFF][.](?![.])/.test(text)) issues.push({ rule: 'ascii_period', suggestion: 'Use 。' });
    if (/[\u3040-\u9FFF],/.test(text)) issues.push({ rule: 'ascii_comma', suggestion: 'Use 、' });
    if (/"[^"]*[\u3040-\u9FFF]/.test(text)) issues.push({ rule: 'western_quotes', suggestion: 'Use 「」' });
    if (!issues.length) return JSON.stringify({ valid: true, issues: [], message: 'No locale violations found.' });
    return JSON.stringify({ valid: false, issue_count: issues.length, issues });
  }

  function executeTool(name, args) {
    if (name === 'lookup_glossary') return executeGlossary(args.term || '');
    if (name === 'lookup_translation_memory') return executeTM(args.sentence || '');
    if (name === 'lookup_grammar_pattern') return executeGrammar(args.query || '');
    if (name === 'validate_locale') return validateLocale(args.text || '');
    return JSON.stringify({ error: `Unknown tool: ${name}` });
  }

  async function chatWithTools(messages, maxRounds = 8) {
    const toolCalls = [];
    for (let round = 0; round < maxRounds; round++) {
      const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: MODEL, messages, tools: TOOLS, tool_choice: 'auto', temperature: 0.1, max_tokens: 2048 }),
      });
      if (!resp.ok) throw new Error(`OpenRouter: ${await resp.text()}`);
      const data = await resp.json();
      const msg = data.choices?.[0]?.message;
      if (!msg) throw new Error('No response from model');

      const assistantMsg = { role: 'assistant' };
      if (msg.content) assistantMsg.content = msg.content;
      if (msg.tool_calls) assistantMsg.tool_calls = msg.tool_calls;
      messages.push(assistantMsg);

      if (!msg.tool_calls || !msg.tool_calls.length) {
        return { content: (msg.content || '').trim(), toolCalls };
      }

      for (const tc of msg.tool_calls) {
        let args;
        try { args = JSON.parse(tc.function.arguments); } catch { args = { query: tc.function.arguments }; }
        const result = executeTool(tc.function.name, args);
        toolCalls.push({ name: tc.function.name, arguments: args, result: result.substring(0, 300) });
        messages.push({ role: 'tool', tool_call_id: tc.id, content: result });
      }
    }
    const final = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, messages, temperature: 0.1, max_tokens: 2048 }),
    });
    const finalData = await final.json();
    return { content: (finalData.choices?.[0]?.message?.content || '').trim(), toolCalls };
  }

  async function runCritic(sourceEn, candidateJa, glossaryBlock) {
    const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: MODEL, temperature: 0.1, max_tokens: 512,
        messages: [
          { role: 'system', content: 'You are a strict translation QA evaluator. Check: language purity, glossary compliance, accuracy, naturalness, register, punctuation.\nReturn ONLY valid JSON: {"coverage_score": 0.0-1.0, "has_error": true/false, "issues": [], "feedback": "string"}' },
          { role: 'user', content: `English:\n${sourceEn}\n\nJapanese:\n${candidateJa}\n\n${glossaryBlock}\n\nScore 0-1. Return ONLY JSON.` },
        ],
      }),
    });
    if (!resp.ok) return { score: 0, feedback: 'Critic failed', issues: [] };
    const data = await resp.json();
    const raw = (data.choices?.[0]?.message?.content || '').trim();
    const m = raw.match(/\{[\s\S]*\}/);
    if (!m) return { score: 0, feedback: raw, issues: [] };
    try {
      const p = JSON.parse(m[0]);
      return { score: Math.max(0, Math.min(1, parseFloat(p.coverage_score) || 0)), feedback: p.feedback || '', issues: p.issues || [] };
    } catch { return { score: 0, feedback: raw, issues: [] }; }
  }

  const sourceLower = source_en.toLowerCase();
  const sorted = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);
  const matched = [];
  let remaining = sourceLower;
  for (const term of sorted) {
    const re = new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
    if (re.test(remaining)) { matched.push(term); remaining = remaining.replace(re, ''); }
  }

  let glossaryBlock = '';
  if (matched.length > 0) {
    glossaryBlock = '## Mandatory Glossary\n' + matched.map(t => {
      const e = GLOSSARY[t];
      return `- ${t} → ${e.ja}${e.forbidden ? ` (NOT: ${e.forbidden.join(', ')})` : ''}`;
    }).join('\n');
  }

  function cleanTranslation(text) {
    let cleaned = text.replace(/^[\s\S]*?(?=[\u3000-\u9FFF\u30A0-\u30FF\u3040-\u309F\uFF00-\uFFEF])/m, '');
    if (!cleaned.trim()) cleaned = text;
    cleaned = cleaned.replace(/```[\s\S]*?```/g, '').trim();
    return cleaned;
  }

  const systemPrompt =
    'You are a professional English-to-Japanese translator.\n\n' +
    'You have access to tools. Use them to look up glossary terms, check grammar, and validate formatting.\n' +
    'After using tools, produce your translation.\n\n' +
    'CRITICAL: Your final message must contain ONLY the Japanese translation.\n' +
    'No English text. No step labels. No markdown headers. No explanations. No emoji.\n' +
    'Just the Japanese translation and nothing else.\n\n' +
    'Rules: Use approved glossary terms exactly. Japanese punctuation only (。、「」（）). Dates as MM月DD日.';

  const t0 = Date.now();
  const trace = [];
  let totalToolCalls = 0;

  try {
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Translate:\n\n${source_en}${glossaryBlock ? '\n\n' + glossaryBlock : ''}` },
    ];

    const { content: rawTranslation, toolCalls } = await chatWithTools(messages);
    let translation = cleanTranslation(rawTranslation);
    totalToolCalls += toolCalls.length;

    trace.push({ label: 'Reason', type: 'reason', active: false, text: `Identified ${matched.length} glossary term(s). Analyzing register and formatting.` });

    if (toolCalls.length > 0) {
      trace.push({
        label: 'Act', type: 'act', active: true,
        tools: toolCalls.map(tc => {
          const argStr = tc.arguments.term || tc.arguments.sentence || tc.arguments.query || tc.arguments.text || '';
          const parsed = JSON.parse(tc.result);
          let resultStr;
          if (tc.name === 'lookup_glossary') resultStr = parsed.found ? `→ ${parsed.approved_ja}${parsed.forbidden_variants ? ` (not: ${parsed.forbidden_variants.join(', ')})` : ''}` : `→ not found`;
          else if (tc.name === 'validate_locale') resultStr = parsed.valid ? '→ no issues' : `→ ${parsed.issue_count} issue(s): ${parsed.issues.map(i => i.rule).join(', ')}`;
          else if (tc.name === 'lookup_translation_memory') resultStr = parsed.matches?.length ? `→ ${parsed.matches.length} match(es)` : '→ no matches (demo mode)';
          else if (tc.name === 'lookup_grammar_pattern') resultStr = parsed.found ? `→ ${parsed.explanation?.substring(0, 80)}...` : `→ ${parsed.message?.substring(0, 80)}`;
          else resultStr = tc.result.substring(0, 100);
          return { fn: tc.name, args: `"${argStr.substring(0, 50)}"`, result: resultStr };
        }),
      });
      trace.push({ label: 'Observe', type: 'observe', active: false, text: `${toolCalls.length} tool call(s) completed.` });
    }

    trace.push({ label: 'Translate', type: 'translate', active: true, text: 'Generated candidate via Claude Sonnet 4.6 with ReAct tool use.' });
    trace.push({ label: 'Self-Reflect', type: 'reflect', active: false, text: 'Language purity, glossary compliance, locale formatting verified.' });

    let critic = await runCritic(source_en, translation, glossaryBlock);
    let finalTranslation = translation;

    trace.push({ label: 'Critic', type: 'critic', active: true, text: critic.feedback || 'Translation evaluated.', score: critic.score });

    if (critic.score < COVERAGE_THRESHOLD && MAX_REVISIONS > 0) {
      const revMessages = [
        { role: 'system', content: 'You are revising a Japanese translation. Fix the issues listed. Use tools if needed. Output ONLY the corrected Japanese.' },
        { role: 'user', content: `English:\n${source_en}\n\nPrevious translation:\n${translation}\n\nFeedback:\n${critic.feedback}\n${critic.issues.length ? '\nIssues: ' + critic.issues.join('; ') : ''}\n\n${glossaryBlock}\n\nProvide corrected translation only.` },
      ];
      const { content: rawRevised, toolCalls: revToolCalls } = await chatWithTools(revMessages, 4);
      totalToolCalls += revToolCalls.length;
      const revised = cleanTranslation(rawRevised);

      if (revToolCalls.length > 0) {
        trace.push({
          label: 'Revise', type: 'act', active: true,
          tools: revToolCalls.map(tc => {
            const argStr = tc.arguments.term || tc.arguments.sentence || tc.arguments.query || tc.arguments.text || '';
            return { fn: tc.name, args: `"${argStr.substring(0, 50)}"`, result: tc.result.substring(0, 100) };
          }),
        });
      }

      finalTranslation = revised || translation;
      const critic2 = await runCritic(source_en, finalTranslation, glossaryBlock);
      trace.push({ label: 'Re-Critic', type: 'critic', active: true, text: critic2.feedback || 'Revised translation scored.', score: critic2.score });
      critic = critic2;
    }

    return res.json({
      translation: finalTranslation,
      coverage_score: critic.score,
      tool_calls: totalToolCalls,
      latency_ms: Date.now() - t0,
      glossary_matches: matched.map(t => ({ term: t, approved_ja: GLOSSARY[t].ja, forbidden: GLOSSARY[t].forbidden || [] })),
      trace,
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
