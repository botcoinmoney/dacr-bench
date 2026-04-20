/**
 * DACR-Bench Evaluation Harness
 *
 * Scores model predictions against gold answers.
 * Includes confidence calibration metrics — models must provide confidence per answer.
 */

import * as fs from "fs";
import type {
  BenchmarkChallenge,
  BenchmarkQuestion,
  BenchmarkSplit,
  ModelPrediction,
  ModelAnswer,
  QuestionScore,
  ChallengeScore,
  BenchmarkResults,
  SplitSummary,
  QuestionCategory,
  ParseFormat,
} from "./types.js";

// ── Answer Matching ──

type MatchType = "exact" | "normalized" | "numeric" | "partial" | "none";

function matchAnswer(modelAnswer: string, question: BenchmarkQuestion): MatchType {
  const normalize = (s: string) =>
    s.toLowerCase().trim()
      .replace(/^(the|a|an)\s+/i, "")
      .replace(/[.,;:!?]+$/, "")
      .replace(/\s+/g, " ")
      .trim();

  const modelNorm = normalize(modelAnswer);
  const goldNorm = normalize(question.goldAnswer);

  // Empty or missing answer is never correct
  if (!modelNorm || !goldNorm) return "none";

  // Exact match
  if (modelNorm === goldNorm) return "exact";

  // Alias match
  if (question.goldAnswerAliases.some((a) => normalize(a) === modelNorm)) return "normalized";

  // Numeric match (within 1%)
  const modelNum = parseFloat(modelAnswer.replace(/[^0-9.\-]/g, ""));
  const goldNum = parseFloat(question.goldAnswer.replace(/[^0-9.\-]/g, ""));
  if (!isNaN(modelNum) && !isNaN(goldNum) && goldNum !== 0) {
    if (Math.abs(modelNum - goldNum) / Math.abs(goldNum) < 0.01) return "numeric";
  }

  // Containment match (model's answer contains gold answer or vice versa)
  if (modelNorm.includes(goldNorm) || goldNorm.includes(modelNorm)) return "partial";

  // Token-level overlap
  const modelTokens = new Set(modelNorm.split(/\s+/).filter((t) => t.length > 0));
  const goldTokens = new Set(goldNorm.split(/\s+/).filter((t) => t.length > 0));
  if (modelTokens.size === 0 || goldTokens.size === 0) return "none";
  const intersection = [...modelTokens].filter((t) => goldTokens.has(t));
  const f1 = (2 * intersection.length) / (modelTokens.size + goldTokens.size);
  if (f1 >= 0.8) return "partial";

  return "none";
}

// ── Citation Scoring ──

function scoreCitation(
  citation: string,
  question: BenchmarkQuestion,
  documentText: string
): number {
  if (!citation || citation.trim().length === 0) return 0;

  const citationLower = citation.toLowerCase();

  // Check if citation references the right section/area
  for (const step of question.reasoningChain) {
    if (step.factId) {
      // Check if citation mentions relevant keywords from the reasoning chain
      if (step.note && citationLower.includes(step.note.toLowerCase().slice(0, 20))) {
        return 1.0;
      }
    }
  }

  // Check if citation text appears near the answer in the document
  const docLower = documentText.toLowerCase();
  const goldLower = question.goldAnswer.toLowerCase();
  const answerIdx = docLower.indexOf(goldLower);
  if (answerIdx >= 0) {
    // Extract surrounding context
    const context = docLower.slice(Math.max(0, answerIdx - 500), answerIdx + 500);
    const citationTokens = citationLower.split(/\s+/).filter((t) => t.length > 3);
    const matchCount = citationTokens.filter((t) => context.includes(t)).length;
    if (matchCount >= citationTokens.length * 0.5) return 1.0;
    if (matchCount >= citationTokens.length * 0.25) return 0.5;
  }

  return 0;
}

// ── Trap Scoring ──

function scoreTrapEvasion(
  modelAnswer: string,
  question: BenchmarkQuestion,
  challenge: BenchmarkChallenge
): boolean | undefined {
  if (!question.targetsTrap) return undefined;

  const trap = challenge.traps.find((t) => t.trapId === question.targetsTrap);
  if (!trap) return undefined;

  const normalize = (s: string) => s.toLowerCase().trim().replace(/[^a-z0-9.]/g, "");
  const modelNorm = normalize(modelAnswer);
  const correctNorm = normalize(String(trap.correctValue));
  const wrongNorm = normalize(String(trap.wrongValue));

  if (modelNorm.includes(correctNorm)) return true;  // avoided trap
  if (modelNorm.includes(wrongNorm)) return false;    // fell for trap
  return undefined; // wrong for other reasons
}

// ── Split Inference ──

/** Determine split from challenge — uses explicit field or infers from source type */
function inferSplit(challenge: BenchmarkChallenge): BenchmarkSplit {
  if ((challenge as any).split) return (challenge as any).split;
  return challenge.source.type === "engine" ? "synthetic" : "real";
}

// ── Confidence Calibration ──

/**
 * Expected Calibration Error (ECE)
 * Measures how well a model's confidence predicts its accuracy.
 * Lower is better. 0 = perfectly calibrated.
 */
function computeECE(scores: QuestionScore[], bins: number = 10): number {
  const binSize = 1 / bins;
  let totalECE = 0;
  const totalSamples = scores.length;

  for (let b = 0; b < bins; b++) {
    const lower = b * binSize;
    const upper = (b + 1) * binSize;
    const binScores = scores.filter(
      (s) => s.confidenceScore >= lower && s.confidenceScore < upper
    );
    if (binScores.length === 0) continue;

    const avgConfidence = binScores.reduce((s, q) => s + q.confidenceScore, 0) / binScores.length;
    const accuracy = binScores.filter((q) => q.correct).length / binScores.length;
    totalECE += (binScores.length / totalSamples) * Math.abs(accuracy - avgConfidence);
  }

  return totalECE;
}

// ── Score a Single Challenge ──

function scoreChallenge(
  challenge: BenchmarkChallenge,
  prediction: ModelPrediction
): ChallengeScore {
  const questionScores: QuestionScore[] = [];

  for (const q of challenge.questions) {
    const modelResponse: ModelAnswer | undefined = prediction.answers[q.questionId];

    if (!modelResponse) {
      questionScores.push({
        questionId: q.questionId,
        category: q.category,
        hops: q.hops,
        correct: false,
        confidenceScore: 0,
        citationScore: 0,
        trapEvasion: q.targetsTrap ? false : undefined,
        matchType: "none",
      });
      continue;
    }

    const matchType = matchAnswer(modelResponse.answer, q);
    const correct = matchType !== "none";
    const citationScore = scoreCitation(modelResponse.citation, q, challenge.document.text);
    const trapEvasion = scoreTrapEvasion(modelResponse.answer, q, challenge);
    const confidence = Math.max(0, Math.min(1, modelResponse.confidence ?? 0));

    questionScores.push({
      questionId: q.questionId,
      category: q.category,
      hops: q.hops,
      correct,
      confidenceScore: confidence,
      citationScore,
      trapEvasion,
      matchType,
    });
  }

  const correctCount = questionScores.filter((s) => s.correct).length;
  const answerAccuracy = correctCount / questionScores.length;

  const trapQuestions = questionScores.filter((s) => s.trapEvasion !== undefined);
  const trapEvasionRate = trapQuestions.length > 0
    ? trapQuestions.filter((s) => s.trapEvasion === true).length / trapQuestions.length
    : 1;

  const citationGroundingScore = questionScores.reduce((s, q) => s + q.citationScore, 0) / questionScores.length;
  const confidenceCalibration = computeECE(questionScores);

  return {
    challengeId: challenge.challengeId,
    answerAccuracy,
    confidenceCalibration,
    citationGroundingScore,
    trapEvasionRate,
    passed: answerAccuracy >= 0.6,
    questionScores,
  };
}

// ── Aggregate Results ──

export function evaluate(
  challenges: BenchmarkChallenge[],
  predictions: ModelPrediction[]
): BenchmarkResults {
  const predMap = new Map(predictions.map((p) => [p.challengeId, p]));
  const challengeScores: ChallengeScore[] = [];
  const allQuestionScores: QuestionScore[] = [];
  let formatFailures = 0;

  // Track format breakdown
  const formatCounts: Record<ParseFormat, { count: number; correct: number; total: number }> = {
    json: { count: 0, correct: 0, total: 0 },
    trace: { count: 0, correct: 0, total: 0 },
    prose: { count: 0, correct: 0, total: 0 },
    failed: { count: 0, correct: 0, total: 0 },
  };

  for (const challenge of challenges) {
    const pred = predMap.get(challenge.challengeId);
    if (!pred) {
      formatFailures++;
      formatCounts.failed.count++;
      continue;
    }

    // Track format used
    const parseFormat: ParseFormat = pred.metadata?.parseFormat ?? "json";
    formatCounts[parseFormat].count++;

    const score = scoreChallenge(challenge, pred);
    challengeScores.push(score);
    allQuestionScores.push(...score.questionScores);

    // Track accuracy by format
    formatCounts[parseFormat].total += score.questionScores.length;
    formatCounts[parseFormat].correct += score.questionScores.filter(q => q.correct).length;
  }

  // Category breakdown
  const byCategory: Record<string, { accuracy: number; meanConfidence: number; count: number }> = {};
  const categories: QuestionCategory[] = [
    "direct_extraction", "multi_hop_bridge", "comparative", "computation",
    "conditional_filtered", "trap_targeted", "cross_section_synthesis",
  ];
  for (const cat of categories) {
    const qs = allQuestionScores.filter((s) => s.category === cat);
    if (qs.length === 0) continue;
    byCategory[cat] = {
      accuracy: qs.filter((q) => q.correct).length / qs.length,
      meanConfidence: qs.reduce((s, q) => s + q.confidenceScore, 0) / qs.length,
      count: qs.length,
    };
  }

  // Domain breakdown
  const byDomain: Record<string, { accuracy: number; meanConfidence: number; count: number }> = {};
  for (const challenge of challenges) {
    const domain = challenge.document.domain;
    if (!byDomain[domain]) byDomain[domain] = { accuracy: 0, meanConfidence: 0, count: 0 };
    const score = challengeScores.find((s) => s.challengeId === challenge.challengeId);
    if (!score) continue;
    const qs = score.questionScores;
    const existing = byDomain[domain];
    const totalCount = existing.count + qs.length;
    existing.accuracy = (existing.accuracy * existing.count + qs.filter((q) => q.correct).length) / totalCount;
    existing.meanConfidence = (existing.meanConfidence * existing.count + qs.reduce((s, q) => s + q.confidenceScore, 0)) / totalCount;
    existing.count = totalCount;
  }

  // Hop breakdown
  const byHops: Record<string, { accuracy: number; meanConfidence: number; count: number }> = {};
  for (const hopLabel of ["1", "2", "3", "4", "5", "6+"]) {
    const qs = allQuestionScores.filter((s) =>
      hopLabel === "6+" ? s.hops >= 6 : s.hops === parseInt(hopLabel)
    );
    if (qs.length === 0) continue;
    byHops[hopLabel] = {
      accuracy: qs.filter((q) => q.correct).length / qs.length,
      meanConfidence: qs.reduce((s, q) => s + q.confidenceScore, 0) / qs.length,
      count: qs.length,
    };
  }

  // Difficulty breakdown
  const byDifficulty: Record<string, { accuracy: number; meanConfidence: number; count: number }> = {};
  for (const diff of ["easy", "medium", "hard"]) {
    const matchingQIds = new Set<string>();
    for (const ch of challenges) {
      for (const q of ch.questions) {
        if (q.difficulty === diff) matchingQIds.add(`${ch.challengeId}:${q.questionId}`);
      }
    }
    const qs = allQuestionScores.filter((s) => {
      const challenge = challenges.find((c) => c.questions.some((q) => q.questionId === s.questionId));
      return challenge && challenge.questions.find((q) => q.questionId === s.questionId)?.difficulty === diff;
    });
    if (qs.length === 0) continue;
    byDifficulty[diff] = {
      accuracy: qs.filter((q) => q.correct).length / qs.length,
      meanConfidence: qs.reduce((s, q) => s + q.confidenceScore, 0) / qs.length,
      count: qs.length,
    };
  }

  // Split breakdown
  const bySplit: Record<string, SplitSummary> = {};
  for (const split of ["real", "synthetic"] as BenchmarkSplit[]) {
    const splitChallenges = challenges.filter((c) => inferSplit(c) === split);
    const splitScores = challengeScores.filter((cs) =>
      splitChallenges.some((c) => c.challengeId === cs.challengeId)
    );
    const splitQs = splitScores.flatMap((s) => s.questionScores);
    if (splitQs.length === 0) continue;

    const splitTrapQs = splitQs.filter((q) => q.trapEvasion !== undefined);
    bySplit[split] = {
      answerAccuracy: splitQs.filter((q) => q.correct).length / splitQs.length,
      meanConfidence: splitQs.reduce((s, q) => s + q.confidenceScore, 0) / splitQs.length,
      trapEvasionRate: splitTrapQs.length > 0
        ? splitTrapQs.filter((q) => q.trapEvasion === true).length / splitTrapQs.length
        : 1,
      passRate: splitScores.length > 0
        ? splitScores.filter((c) => c.passed).length / splitScores.length
        : 0,
      challengeCount: splitScores.length,
      questionCount: splitQs.length,
    };
  }

  // Transfer gap: |AA_real - AA_synthetic|
  const transferGap = (bySplit.real && bySplit.synthetic)
    ? Math.abs(bySplit.real.answerAccuracy - bySplit.synthetic.answerAccuracy)
    : undefined;

  // Summary
  const totalQuestions = allQuestionScores.length;
  const totalCorrect = allQuestionScores.filter((q) => q.correct).length;
  const totalConfidence = allQuestionScores.reduce((s, q) => s + q.confidenceScore, 0);
  const confidenceWeightedCorrect = allQuestionScores.reduce(
    (s, q) => s + (q.correct ? q.confidenceScore : 0), 0
  );

  const trapQs = allQuestionScores.filter((s) => s.trapEvasion !== undefined);

  const modelName = predictions[0]?.model ?? "unknown";

  return {
    model: modelName,
    benchmarkVersion: challenges[0]?.version ?? "1.0",
    timestamp: new Date().toISOString(),
    summary: {
      answerAccuracy: totalQuestions > 0 ? totalCorrect / totalQuestions : 0,
      confidenceWeightedAccuracy: totalConfidence > 0 ? confidenceWeightedCorrect / totalConfidence : 0,
      confidenceCalibration: computeECE(allQuestionScores),
      meanConfidence: totalQuestions > 0 ? totalConfidence / totalQuestions : 0,
      trapEvasionRate: trapQs.length > 0
        ? trapQs.filter((q) => q.trapEvasion === true).length / trapQs.length
        : 1,
      citationGroundingScore: totalQuestions > 0
        ? allQuestionScores.reduce((s, q) => s + q.citationScore, 0) / totalQuestions
        : 0,
      passRate: challengeScores.length > 0
        ? challengeScores.filter((c) => c.passed).length / challengeScores.length
        : 0,
      challengesEvaluated: challengeScores.length,
      questionsEvaluated: totalQuestions,
      formatFailureRate: challenges.length > 0 ? formatFailures / challenges.length : 0,
    },
    byCategory: byCategory as any,
    byDomain,
    byHops,
    byDifficulty,
    bySplit: Object.keys(bySplit).length > 0 ? bySplit as any : undefined,
    byFormat: {
      json: {
        count: formatCounts.json.count,
        accuracy: formatCounts.json.total > 0 ? formatCounts.json.correct / formatCounts.json.total : 0,
        total: formatCounts.json.total,
      },
      trace: {
        count: formatCounts.trace.count,
        accuracy: formatCounts.trace.total > 0 ? formatCounts.trace.correct / formatCounts.trace.total : 0,
        total: formatCounts.trace.total,
      },
      prose: {
        count: formatCounts.prose.count,
        accuracy: formatCounts.prose.total > 0 ? formatCounts.prose.correct / formatCounts.prose.total : 0,
        total: formatCounts.prose.total,
      },
      failed: {
        count: formatCounts.failed.count,
        accuracy: 0,
        total: 0,
      },
    },
    transferGap,
    challengeScores,
  };
}

// ── Report Generation ──

export function generateReport(results: BenchmarkResults): string {
  const s = results.summary;
  let report = `# DACR-Bench Results: ${results.model}\n\n`;
  report += `**Benchmark Version:** ${results.benchmarkVersion}\n`;
  report += `**Evaluated:** ${s.challengesEvaluated} challenges, ${s.questionsEvaluated} questions\n`;
  report += `**Timestamp:** ${results.timestamp}\n\n`;

  report += `## Summary\n\n`;
  report += `| Metric | Score |\n|---|---|\n`;
  report += `| Answer Accuracy | ${(s.answerAccuracy * 100).toFixed(1)}% |\n`;
  report += `| Confidence-Weighted Accuracy | ${(s.confidenceWeightedAccuracy * 100).toFixed(1)}% |\n`;
  report += `| Confidence Calibration (ECE) | ${s.confidenceCalibration.toFixed(3)} |\n`;
  report += `| Mean Confidence | ${(s.meanConfidence * 100).toFixed(1)}% |\n`;
  report += `| Trap Evasion Rate | ${(s.trapEvasionRate * 100).toFixed(1)}% |\n`;
  report += `| Citation Grounding | ${(s.citationGroundingScore * 100).toFixed(1)}% |\n`;
  report += `| Challenge Pass Rate | ${(s.passRate * 100).toFixed(1)}% |\n`;
  report += `| Format Failure Rate | ${(s.formatFailureRate * 100).toFixed(1)}% |\n\n`;

  // Format breakdown
  if (results.byFormat) {
    report += `## Response Format Breakdown\n\n`;
    report += `*Shows how answers were extracted from model responses*\n\n`;
    report += `| Format | Challenges | Questions | Accuracy |\n|---|---|---|---|\n`;
    for (const [fmt, data] of Object.entries(results.byFormat)) {
      if (data.count > 0) {
        report += `| ${fmt.toUpperCase()} | ${data.count} | ${data.total} | ${(data.accuracy * 100).toFixed(1)}% |\n`;
      }
    }
    report += `\n`;
  }

  report += `## By Category\n\n`;
  report += `| Category | Accuracy | Confidence | Count |\n|---|---|---|---|\n`;
  for (const [cat, data] of Object.entries(results.byCategory)) {
    report += `| ${cat} | ${(data.accuracy * 100).toFixed(1)}% | ${(data.meanConfidence * 100).toFixed(1)}% | ${data.count} |\n`;
  }

  report += `\n## By Hop Depth\n\n`;
  report += `| Hops | Accuracy | Confidence | Count |\n|---|---|---|---|\n`;
  for (const [hop, data] of Object.entries(results.byHops)) {
    report += `| ${hop} | ${(data.accuracy * 100).toFixed(1)}% | ${(data.meanConfidence * 100).toFixed(1)}% | ${data.count} |\n`;
  }

  report += `\n## By Domain\n\n`;
  report += `| Domain | Accuracy | Confidence | Count |\n|---|---|---|---|\n`;
  for (const [domain, data] of Object.entries(results.byDomain)) {
    report += `| ${domain} | ${(data.accuracy * 100).toFixed(1)}% | ${(data.meanConfidence * 100).toFixed(1)}% | ${data.count} |\n`;
  }

  report += `\n## By Difficulty\n\n`;
  report += `| Difficulty | Accuracy | Confidence | Count |\n|---|---|---|---|\n`;
  for (const [diff, data] of Object.entries(results.byDifficulty)) {
    report += `| ${diff} | ${(data.accuracy * 100).toFixed(1)}% | ${(data.meanConfidence * 100).toFixed(1)}% | ${data.count} |\n`;
  }

  if (results.bySplit && Object.keys(results.bySplit).length > 0) {
    report += `\n## By Split (Real vs Synthetic)\n\n`;
    report += `| Split | Accuracy | Confidence | Trap Evasion | Pass Rate | Challenges | Questions |\n|---|---|---|---|---|---|---|\n`;
    for (const [split, data] of Object.entries(results.bySplit)) {
      report += `| ${split} | ${(data.answerAccuracy * 100).toFixed(1)}% | ${(data.meanConfidence * 100).toFixed(1)}% | ${(data.trapEvasionRate * 100).toFixed(1)}% | ${(data.passRate * 100).toFixed(1)}% | ${data.challengeCount} | ${data.questionCount} |\n`;
    }
    if (results.transferGap !== undefined) {
      report += `\n**Transfer Gap:** ${(results.transferGap * 100).toFixed(1)}pp`;
      report += ` (|AA_real - AA_synthetic| — lower means synthetic performance predicts real-doc performance)\n`;
    }
  }

  return report;
}

// ── CLI Entry Point ──

if (process.argv[1]?.endsWith("evaluate.ts")) {
  const args = process.argv.slice(2);
  const benchIdx = args.indexOf("--benchmark");
  const predIdx = args.indexOf("--predictions");
  const outIdx = args.indexOf("--output");
  const splitIdx = args.indexOf("--split");

  if (benchIdx === -1 || predIdx === -1) {
    console.error("Usage: npx tsx src/evaluate.ts --benchmark <challenges.json> --predictions <predictions.json> [--output <results.json>] [--split real|synthetic|all]");
    process.exit(1);
  }

  let challenges: BenchmarkChallenge[] = JSON.parse(fs.readFileSync(args[benchIdx + 1], "utf-8"));

  // Handle BenchmarkDataset wrapper (has .challenges array) vs raw array
  if (!Array.isArray(challenges) && (challenges as any).challenges) {
    challenges = (challenges as any).challenges;
  }

  const predictions: ModelPrediction[] = JSON.parse(fs.readFileSync(args[predIdx + 1], "utf-8"));
  const outputPath = outIdx !== -1 ? args[outIdx + 1] : "results/results.json";
  const splitFilter = splitIdx !== -1 ? args[splitIdx + 1] as "real" | "synthetic" | "all" : "all";

  // Filter by split if requested
  if (splitFilter !== "all") {
    challenges = challenges.filter((c) => inferSplit(c) === splitFilter);
    console.log(`Filtered to ${splitFilter} split: ${challenges.length} challenges\n`);
  }

  const results = evaluate(challenges, predictions);
  const report = generateReport(results);

  fs.mkdirSync("results", { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  fs.writeFileSync(outputPath.replace(".json", ".md"), report);

  console.log(report);
  console.log(`\nResults written to ${outputPath}`);
}
