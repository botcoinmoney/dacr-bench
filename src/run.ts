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
  timeoutMs: number
): string {
  const payload = JSON.stringify({
    model,
    messages: [{ role: "user", content: prompt }],
    temperature,
    max_tokens: 2000,
  });

  const tmpPayload = `/tmp/dacr_payload_${crypto.randomBytes(4).toString("hex")}.json`;
  fs.writeFileSync(tmpPayload, payload);

  try {
    const result = execSync(
      `curl -s -X POST "${endpoint}" -H "Content-Type: application/json" -d @"${tmpPayload}" --max-time ${Math.ceil(timeoutMs / 1000)}`,
      { maxBuffer: 10 * 1024 * 1024, timeout: timeoutMs + 5000 }
    );

    const parsed = JSON.parse(result.toString());
    // OpenAI-compatible format
    if (parsed.choices?.[0]?.message?.content) {
      return parsed.choices[0].message.content;
    }
    // Anthropic format
    if (parsed.content?.[0]?.text) {
      return parsed.content[0].text;
    }
    return result.toString();
  } finally {
    try { fs.unlinkSync(tmpPayload); } catch {}
  }
}

// ── Response Parsing ──

function parseResponse(
  raw: string,
  challenge: BenchmarkChallenge
): Record<string, ModelAnswer> {
  // Strip markdown code fences
  let cleaned = raw.replace(/```json\s*/g, "").replace(/```\s*/g, "");

  // Extract first JSON object
  const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error("No JSON object found in response");
  }

  let parsed: Record<string, any>;
  try {
    parsed = JSON.parse(jsonMatch[0]);
  } catch {
    // Try to fix common JSON issues
    const fixed = jsonMatch[0]
      .replace(/'/g, '"')
      .replace(/,\s*}/g, "}")
      .replace(/,\s*]/g, "]");
    parsed = JSON.parse(fixed);
  }

  const answers: Record<string, ModelAnswer> = {};
  for (const q of challenge.questions) {
    const entry = parsed[q.questionId];
    if (!entry) {
      answers[q.questionId] = { answer: "", citation: "", confidence: 0 };
      continue;
    }

    answers[q.questionId] = {
      answer: String(entry.answer ?? entry.Answer ?? ""),
      citation: String(entry.citation ?? entry.Citation ?? ""),
      confidence: Math.max(0, Math.min(1, parseFloat(entry.confidence ?? entry.Confidence ?? "0") || 0)),
    };
  }

  return answers;
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
        raw = runHTTP(prompt, config.endpoint!, config.model, temperature, timeoutMs);
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
    try {
      answers = parseResponse(raw, challenge);
    } catch (e: any) {
      console.warn(`  PARSE FAILED: ${e.message}`);
      answers = {};
      formatFailures = 1;
    }

    const answeredCount = Object.values(answers).filter((a) => a.answer).length;
    const avgConfidence = answeredCount > 0
      ? Object.values(answers).reduce((s, a) => s + a.confidence, 0) / answeredCount
      : 0;

    console.log(`  ${answeredCount}/${challenge.questions.length} answered, avg confidence: ${(avgConfidence * 100).toFixed(0)}%, ${elapsedSeconds.toFixed(1)}s`);

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
    temperature,
    timeoutSeconds: timeout,
    outputPath,
    limit,
    split,
  });
}
