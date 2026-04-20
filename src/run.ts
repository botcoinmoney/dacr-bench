/**
 * DACR-Bench Model Runner
 *
 * Runs models against benchmark challenges and collects predictions.
 * Supports Claude Code CLI models (via claude -p) and local model servers (via HTTP).
 *
 * IMPORTANT: The evaluation prompt does NOT warn about conflicting information.
 * This tests raw reasoning ability without hints about traps.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { execSync } from "child_process";
import type { BenchmarkChallenge, BenchmarkSplit, ModelPrediction, ModelAnswer } from "./types.js";

// ── Prompt Template ──
// NOTE: No conflicting information warning — deliberately removed per benchmark design

const PROMPT_TEMPLATE = `You are given a technical document and a set of questions. For each question, find the answer in the document and state where you found it.

For each answer, also provide a confidence score between 0.0 and 1.0 indicating how confident you are in your answer, where 0.0 means pure guess and 1.0 means absolutely certain.

Document:
{document}

Questions:
{questions}

Respond with ONLY a JSON object. No explanation, no reasoning, no preamble.
For each question, provide the answer, a brief citation, and your confidence score.

{
  "q01": {"answer": "...", "citation": "...", "confidence": 0.95},
  "q02": {"answer": "...", "citation": "...", "confidence": 0.8}
}`;

// ── Configuration ──

export interface RunConfig {
  /** Path to benchmark challenges JSON */
  benchmarkPath: string;
  /** Model identifier */
  model: string;
  /** Runner type */
  runner: "claude-cli" | "http";
  /** HTTP endpoint for local models (only for runner=http) */
  endpoint?: string;
  /** API key for authenticated endpoints */
  apiKey?: string;
  /** Temperature for inference */
  temperature?: number;
  /** Timeout per challenge in seconds */
  timeoutSeconds?: number;
  /** Output path for predictions */
  outputPath: string;
  /** Max challenges to run (for quick testing) */
  limit?: number;
  /** Filter to specific split */
  split?: BenchmarkSplit | "all";
}

// ── Prompt Formatting ──

function formatPrompt(challenge: BenchmarkChallenge): string {
  const questionList = challenge.questions
    .map((q, i) => `${i + 1}. [${q.questionId}] ${q.text}`)
    .join("\n");

  return PROMPT_TEMPLATE
    .replace("{document}", challenge.document.text)
    .replace("{questions}", questionList);
}

// ── Claude CLI Runner ──

function runClaudeCLI(
  prompt: string,
  model: string,
  timeoutMs: number
): string {
  const tmpFile = `/tmp/dacr_run_${crypto.randomBytes(4).toString("hex")}.txt`;
  fs.writeFileSync(tmpFile, prompt);

  try {
    const result = execSync(
      `cat "${tmpFile}" | claude -p --model ${model}`,
      {
        maxBuffer: 50 * 1024 * 1024,
        timeout: timeoutMs,
        cwd: "/tmp", // isolated from repo
        shell: "/bin/bash",
      }
    );
    return result.toString().trim();
  } finally {
    try { fs.unlinkSync(tmpFile); } catch {}
  }
}

// ── HTTP Runner (for local models / vLLM / ollama) ──

function runHTTP(
  prompt: string,
  endpoint: string,
  model: string,
  temperature: number,
  timeoutMs: number,
  apiKey?: string
): string {
  const payload = JSON.stringify({
    model,
    messages: [{ role: "user", content: prompt }],
    temperature,
    max_tokens: 4000,
  });

  const tmpPayload = `/tmp/dacr_payload_${crypto.randomBytes(4).toString("hex")}.json`;
  fs.writeFileSync(tmpPayload, payload);

  const maxRetries = 3;

  try {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      const result = execSync(
        `curl -s -X POST "${endpoint}" -H "Content-Type: application/json"${apiKey ? ` -H "Authorization: Bearer ${apiKey}"` : ""} -d @"${tmpPayload}" --max-time ${Math.ceil(timeoutMs / 1000)}`,
        { maxBuffer: 10 * 1024 * 1024, timeout: timeoutMs + 5000 }
      );

      let responseBody: any;
      try {
        responseBody = JSON.parse(result.toString());
      } catch {
        // Response isn't JSON — could be raw text from some endpoints
        return result.toString().trim();
      }

      // Handle array-wrapped responses (some APIs wrap in [])
      if (Array.isArray(responseBody)) {
        responseBody = responseBody[0];
      }

      // Check for rate-limit / quota errors and retry with backoff
      if (responseBody.error) {
        const code = responseBody.error.code ?? responseBody.error.status;
        if ((code === 429 || code === "RESOURCE_EXHAUSTED") && attempt < maxRetries - 1) {
          const retryMatch = JSON.stringify(responseBody).match(/retry.*?(\d+)/i);
          const waitSecs = retryMatch ? Math.min(parseInt(retryMatch[1]) + 5, 120) : 30 * (attempt + 1);
          console.warn(`  Rate limited, waiting ${waitSecs}s before retry ${attempt + 2}/${maxRetries}...`);
          execSync(`sleep ${waitSecs}`);
          continue;
        }
        throw new Error(`API error: ${responseBody.error.message ?? JSON.stringify(responseBody.error)}`);
      }

      // OpenAI-compatible format
      if (responseBody.choices?.[0]?.message?.content) {
        return responseBody.choices[0].message.content;
      }
      // Anthropic format
      if (responseBody.content?.[0]?.text) {
        return responseBody.content[0].text;
      }
      return result.toString();
    }
    throw new Error("Max retries exceeded");
  } finally {
    try { fs.unlinkSync(tmpPayload); } catch {}
  }
}

// ── Response Parsing ──

/**
 * Attempt to parse JSON response (strict mode).
 * Returns null if no valid JSON found.
 */
function tryParseJSON(raw: string): Record<string, any> | null {
  let cleaned = raw.replace(/```json\s*/g, "").replace(/```\s*/g, "");

  const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return null;

  try {
    return JSON.parse(jsonMatch[0]);
  } catch {
    // Try to fix common JSON issues
    const fixed = jsonMatch[0]
      .replace(/'/g, '"')
      .replace(/,\s*}/g, "}")
      .replace(/,\s*]/g, "]");
    try {
      return JSON.parse(fixed);
    } catch {
      return null;
    }
  }
}

/**
 * Detect API error responses that masquerade as valid JSON.
 * Returns true if the response is an error, not a model answer.
 */
function isErrorResponse(parsed: Record<string, any>): boolean {
  return !!parsed.error || !!parsed.message && !!parsed.code;
}

/**
 * Try to extract answers from DACR trace format with <|artifact|> block.
 * Returns null if no artifact found.
 */
function tryParseTrace(raw: string, challenge: BenchmarkChallenge): Record<string, ModelAnswer> | null {
  // Look for <|artifact|> block
  const artifactMatch = raw.match(/<\|artifact\|>\s*([\s\S]*?)(?:<\|end|$)/);
  if (!artifactMatch) return null;

  const artifactText = artifactMatch[0].replace(/<\|artifact\|>\s*/, "").replace(/<\|end.*$/, "").trim();
  if (!artifactText) return null;

  // Extract citations from trace lines (> extract: ... | paragraph_N)
  const citations: string[] = [];
  const extractMatches = raw.matchAll(/>\s*extract:\s*[^|]+\|[^|]+\|[^|]+\|\s*(paragraph_\d+)/gi);
  for (const m of extractMatches) {
    citations.push(m[1]);
  }
  const citationStr = citations.slice(0, 3).join(", ") || "trace-derived";

  const answers: Record<string, ModelAnswer> = {};

  if (challenge.questions.length === 1) {
    // Single question - entire artifact is the answer
    answers[challenge.questions[0].questionId] = {
      answer: artifactText,
      citation: citationStr,
      confidence: 0.7,
    };
    return answers;
  }

  // Multi-question - try to parse artifact structure
  const lines = artifactText.split(/\n/).filter(l => l.trim());

  for (const q of challenge.questions) {
    const qNum = q.questionId.replace(/\D/g, "");
    const patterns = [
      new RegExp(`(?:q|Q|question)?\\s*${qNum}\\s*[:=]\\s*(.+)`, "i"),
    ];

    let found = false;
    for (const line of lines) {
      for (const pattern of patterns) {
        const match = line.match(pattern);
        if (match) {
          answers[q.questionId] = {
            answer: match[1].trim(),
            citation: citationStr,
            confidence: 0.6,
          };
          found = true;
          break;
        }
      }
      if (found) break;
    }

    if (!found) {
      // Use positional mapping
      const idx = challenge.questions.indexOf(q);
      if (idx < lines.length) {
        answers[q.questionId] = {
          answer: lines[idx].trim(),
          citation: citationStr,
          confidence: 0.5,
        };
      } else {
        answers[q.questionId] = {
          answer: artifactText,
          citation: citationStr,
          confidence: 0.3,
        };
      }
    }
  }

  return answers;
}

/**
 * Flexible answer extraction from unstructured text.
 * Looks for patterns like "q01: answer" or "1. answer" or numbered responses.
 * Falls back to extracting answer-like content per question.
 */
function extractFromProse(
  raw: string,
  challenge: BenchmarkChallenge
): Record<string, ModelAnswer> {
  const answers: Record<string, ModelAnswer> = {};
  const lines = raw.split("\n");

  for (const q of challenge.questions) {
    const qIdx = parseInt(q.questionId.replace("q", ""));
    let answer = "";
    let citation = "";
    let confidence = 0.5; // default confidence for unstructured responses

    // Strategy 1: Look for "q01:" or "[q01]" pattern
    const qPattern = new RegExp(`(?:${q.questionId}|\\[${q.questionId}\\])\\s*[:\\-]\\s*(.+)`, "i");
    for (const line of lines) {
      const match = line.match(qPattern);
      if (match) {
        answer = match[1].trim();
        break;
      }
    }

    // Strategy 2: Look for numbered pattern "1." or "1)" matching question index
    if (!answer) {
      const numPattern = new RegExp(`^\\s*${qIdx}[.)\\]]\\s*(.+)`, "m");
      const numMatch = raw.match(numPattern);
      if (numMatch) {
        answer = numMatch[1].trim();
      }
    }

    // Strategy 3: Look for the question text echoed back followed by an answer
    if (!answer) {
      const questionWords = q.text.split(/\s+/).slice(0, 5).join("\\s+");
      const echoPattern = new RegExp(questionWords + "[^\\n]*\\n+\\s*(?:Answer:\\s*)?(.+)", "i");
      const echoMatch = raw.match(echoPattern);
      if (echoMatch) {
        answer = echoMatch[1].trim();
      }
    }

    // Extract citation if present near the answer
    if (answer) {
      const citMatch = answer.match(/\(([^)]+(?:section|paragraph|page|table|figure)[^)]*)\)/i)
        || answer.match(/(?:citation|source|ref|found in)[:\s]+(.+?)(?:\.|$)/i);
      if (citMatch) {
        citation = citMatch[1].trim();
        // Remove citation from answer
        answer = answer.replace(citMatch[0], "").trim();
      }
    }

    // Extract confidence if stated
    const confMatch = raw.match(new RegExp(`${q.questionId}[^}]*confidence[:\\s]+([0-9.]+)`, "i"));
    if (confMatch) {
      confidence = Math.max(0, Math.min(1, parseFloat(confMatch[1]) || 0.5));
    }

    answers[q.questionId] = { answer, citation, confidence };
  }

  return answers;
}

type ParseFormat = "json" | "trace" | "prose" | "failed";

interface ParseResult {
  answers: Record<string, ModelAnswer>;
  format: ParseFormat;
}

/**
 * Parse model response with multi-strategy extraction.
 * Priority: JSON > DACR trace > prose extraction
 */
function parseResponse(
  raw: string,
  challenge: BenchmarkChallenge
): ParseResult {
  // Strategy 1: Try strict JSON parse
  const parsed = tryParseJSON(raw);

  if (parsed && !isErrorResponse(parsed)) {
    // Check if the JSON has the expected question keys
    const answers: Record<string, ModelAnswer> = {};
    let foundAny = false;

    for (const q of challenge.questions) {
      // Try exact key, then numeric key, then index-based key
      const entry = parsed[q.questionId]
        ?? parsed[q.questionId.toUpperCase()]
        ?? parsed[q.questionId.replace("q", "Q")]
        ?? parsed[String(parseInt(q.questionId.replace("q", "")))]
        ?? parsed[`question_${q.questionId.replace("q", "")}`];

      if (!entry) {
        answers[q.questionId] = { answer: "", citation: "", confidence: 0 };
        continue;
      }

      foundAny = true;
      answers[q.questionId] = {
        answer: String(entry.answer ?? entry.Answer ?? entry.response ?? entry.Response ?? ""),
        citation: String(entry.citation ?? entry.Citation ?? entry.source ?? entry.Source ?? entry.reference ?? ""),
        confidence: Math.max(0, Math.min(1, parseFloat(entry.confidence ?? entry.Confidence ?? "0.5") || 0.5)),
      };
    }

    if (foundAny) return { answers, format: "json" };
  }

  // Strategy 2: DACR trace format with <|artifact|>
  const traceAnswers = tryParseTrace(raw, challenge);
  if (traceAnswers) {
    const traceFound = Object.values(traceAnswers).filter((a) => a.answer).length;
    if (traceFound > 0) return { answers: traceAnswers, format: "trace" };
  }

  // Strategy 3: Prose/freeform extraction
  const proseAnswers = extractFromProse(raw, challenge);
  const proseFound = Object.values(proseAnswers).filter((a) => a.answer).length;

  if (proseFound > 0) return { answers: proseAnswers, format: "prose" };

  // Strategy 4: Nothing worked
  const emptyAnswers: Record<string, ModelAnswer> = {};
  for (const q of challenge.questions) {
    emptyAnswers[q.questionId] = { answer: "", citation: "", confidence: 0 };
  }
  return { answers: emptyAnswers, format: "failed" };
}

// ── Main Runner ──

/** Infer split from challenge source type if not explicitly set */
function inferSplit(challenge: BenchmarkChallenge): BenchmarkSplit {
  if ((challenge as any).split) return (challenge as any).split;
  return challenge.source.type === "engine" ? "synthetic" : "real";
}

export async function runBenchmark(config: RunConfig): Promise<ModelPrediction[]> {
  let rawData: any = JSON.parse(fs.readFileSync(config.benchmarkPath, "utf-8"));
  let challenges: BenchmarkChallenge[] = Array.isArray(rawData) ? rawData : rawData.challenges ?? rawData;

  // Filter by split
  if (config.split && config.split !== "all") {
    challenges = challenges.filter((c) => inferSplit(c) === config.split);
    console.log(`Filtered to ${config.split} split: ${challenges.length} challenges`);
  }

  const limit = config.limit ?? challenges.length;
  const temperature = config.temperature ?? 0.1;
  const timeoutMs = (config.timeoutSeconds ?? 120) * 1000;
  const predictions: ModelPrediction[] = [];

  console.log(`Running ${Math.min(limit, challenges.length)} challenges with model: ${config.model}`);
  console.log(`Runner: ${config.runner}, Temperature: ${temperature}\n`);

  for (let i = 0; i < Math.min(limit, challenges.length); i++) {
    const challenge = challenges[i];
    console.log(`[${i + 1}/${Math.min(limit, challenges.length)}] ${challenge.challengeId}...`);

    const prompt = formatPrompt(challenge);
    const startTime = Date.now();

    let raw: string;
    try {
      if (config.runner === "claude-cli") {
        raw = runClaudeCLI(prompt, config.model, timeoutMs);
      } else {
        raw = runHTTP(prompt, config.endpoint!, config.model, temperature, timeoutMs, config.apiKey);
      }
    } catch (e: any) {
      console.warn(`  FAILED: ${e.message}`);
      predictions.push({
        challengeId: challenge.challengeId,
        model: config.model,
        timestamp: new Date().toISOString(),
        answers: {},
        metadata: {
          inferenceTimeSeconds: (Date.now() - startTime) / 1000,
          tokensGenerated: 0,
          temperature,
          formatFailures: 1,
        },
      });
      continue;
    }

    const elapsedSeconds = (Date.now() - startTime) / 1000;

    let answers: Record<string, ModelAnswer>;
    let formatFailures = 0;
    let parseFormat: ParseFormat = "failed";
    try {
      const result = parseResponse(raw, challenge);
      answers = result.answers;
      parseFormat = result.format;
      if (parseFormat === "failed") {
        formatFailures = 1;
      }
    } catch (e: any) {
      console.warn(`  PARSE FAILED: ${e.message}`);
      answers = {};
      formatFailures = 1;
    }

    const answeredCount = Object.values(answers).filter((a) => a.answer).length;
    const avgConfidence = answeredCount > 0
      ? Object.values(answers).reduce((s, a) => s + a.confidence, 0) / answeredCount
      : 0;

    const formatLabel = parseFormat === "json" ? "JSON" : parseFormat === "trace" ? "TRACE" : parseFormat === "prose" ? "PROSE" : "FAIL";
    console.log(`  ${answeredCount}/${challenge.questions.length} answered [${formatLabel}], avg confidence: ${(avgConfidence * 100).toFixed(0)}%, ${elapsedSeconds.toFixed(1)}s`);

    predictions.push({
      challengeId: challenge.challengeId,
      model: config.model,
      timestamp: new Date().toISOString(),
      answers,
      metadata: {
        inferenceTimeSeconds: elapsedSeconds,
        tokensGenerated: raw.length, // approximate
        temperature,
        formatFailures,
        parseFormat,
      },
    });
  }

  // Save predictions
  fs.mkdirSync(path.dirname(config.outputPath), { recursive: true });
  fs.writeFileSync(config.outputPath, JSON.stringify(predictions, null, 2));
  console.log(`\nPredictions saved to ${config.outputPath}`);

  return predictions;
}

// ── CLI Entry Point ──

if (process.argv[1]?.endsWith("run.ts")) {
  const args = process.argv.slice(2);

  const get = (flag: string): string | undefined => {
    const idx = args.indexOf(flag);
    return idx !== -1 ? args[idx + 1] : undefined;
  };

  const benchmarkPath = get("--benchmark");
  const model = get("--model") ?? "sonnet";
  const runner = (get("--runner") ?? "claude-cli") as "claude-cli" | "http";
  const endpoint = get("--endpoint") ?? "http://localhost:8000/v1/chat/completions";
  const temperature = parseFloat(get("--temperature") ?? "0.1");
  const timeout = parseInt(get("--timeout") ?? "120");
  const limit = get("--limit") ? parseInt(get("--limit")!) : undefined;
  const split = (get("--split") ?? "all") as "real" | "synthetic" | "all";
  const apiKey = get("--api-key");
  const outputPath = get("--output") ?? `results/predictions_${model.replace(/\//g, "_")}.json`;

  if (!benchmarkPath) {
    console.error(`Usage: npx tsx src/run.ts --benchmark <path> [options]

Options:
  --model <name>       Model identifier (default: sonnet)
  --runner <type>      claude-cli or http (default: claude-cli)
  --endpoint <url>     HTTP endpoint for local models
  --temperature <n>    Inference temperature (default: 0.1)
  --timeout <secs>     Timeout per challenge (default: 120)
  --limit <n>          Max challenges to run
  --split <type>       real, synthetic, or all (default: all)
  --output <path>      Output predictions file`);
    process.exit(1);
  }

  runBenchmark({
    benchmarkPath,
    model,
    runner,
    endpoint,
    apiKey,
    temperature,
    timeoutSeconds: timeout,
    outputPath,
    limit,
    split,
  });
}
