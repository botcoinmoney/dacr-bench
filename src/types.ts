/**
 * DACR-Bench Type Definitions
 *
 * Types for the Domain-Agnostic Causal Reasoning Benchmark.
 * Isolated from the main domain pipeline — imports engine types read-only.
 */

// ── Challenge Sources ──

export type ChallengeSource =
  | { type: "engine"; domain: string; seed: bigint }
  | { type: "document"; documentId: string; sourceUrl?: string };

// ── Fact Extraction (for document-based challenges) ──

export interface ExtractedFact {
  factId: string;
  entity: string;
  attribute: string;
  value: string | number;
  valueType: "numerical" | "categorical" | "relational" | "temporal";
  location: {
    section: string;
    paragraphIndex: number;
    contextQuote: string;
  };
}

// ── Trap Definition ──

export interface TrapDef {
  trapId: string;
  targetFactId: string;
  correctValue: string | number;
  wrongValue: string | number;
  trapLocation: {
    section: string;
    paragraphIndex: number;
    injectedText: string;
  };
}

// ── Question Categories ──

export type QuestionCategory =
  | "direct_extraction"
  | "multi_hop_bridge"
  | "comparative"
  | "computation"
  | "conditional_filtered"
  | "trap_targeted"
  | "cross_section_synthesis";

// ── Reasoning Step (gold chain for verification) ──

export interface ReasoningStep {
  step: number;
  action: "extract" | "compute" | "compare" | "filter";
  factId?: string;
  formula?: string;
  result?: string;
  note?: string;
}

// ── Benchmark Question ──

export interface BenchmarkQuestion {
  questionId: string;
  text: string;
  category: QuestionCategory;
  hops: number;
  targetsTrap: string | null; // trapId or null
  goldAnswer: string;
  goldAnswerAliases: string[];
  reasoningChain: ReasoningStep[];
  difficulty: "easy" | "medium" | "hard";
}

// ── Split Label ──

export type BenchmarkSplit = "real" | "synthetic";

// ── Benchmark Challenge (one document + questions) ──

export interface BenchmarkChallenge {
  challengeId: string;
  version: string;
  source: ChallengeSource;
  split: BenchmarkSplit;
  document: {
    text: string;
    domain: string;
    wordCount: number;
    paragraphCount: number;
    sections: string[];
    sourceUrl?: string;
  };
  facts: ExtractedFact[];
  traps: TrapDef[];
  questions: BenchmarkQuestion[];
  metadata: {
    generatedAt: string;
    verificationPasses: number;
    verificationMethod: string;
  };
}

// ── Model Output (what the model must produce) ──

export interface ModelAnswer {
  answer: string;
  citation: string;
  confidence: number; // 0.0 to 1.0 — required
}

export type ParseFormat = "json" | "trace" | "prose" | "failed";

export interface ModelPrediction {
  challengeId: string;
  model: string;
  timestamp: string;
  answers: Record<string, ModelAnswer>; // keyed by questionId
  metadata: {
    inferenceTimeSeconds: number;
    tokensGenerated: number;
    temperature: number;
    formatFailures: number;
    parseFormat?: ParseFormat; // Which format was successfully parsed
  };
}

// ── Verification Result ──

export interface VerificationResult {
  challengeId: string;
  passed: boolean;
  passCount: number;       // how many LLM passes answered correctly
  totalPasses: number;     // total LLM verification attempts
  perQuestion: {
    questionId: string;
    answerable: boolean;   // was the question answerable from the document?
    consistentAnswer: boolean; // did all passes agree on the answer?
    matchesGold: boolean;  // does the consensus answer match gold?
    requiresStatedHops: boolean; // verified multi-hop requirement
    failureReason?: string;
  }[];
  deterministicChecks: {
    allFactsInDocument: boolean;
    noAmbiguousQuestions: boolean;
    trapValuesPresent: boolean;
    goldAnswersExtractable: boolean;
    computationsVerified: boolean;
  };
}

// ── Evaluation Scores ──

export interface QuestionScore {
  questionId: string;
  category: QuestionCategory;
  hops: number;
  correct: boolean;
  confidenceScore: number;    // model's stated confidence
  citationScore: number;      // 0, 0.5, or 1.0
  trapEvasion?: boolean;      // only for trap-targeted questions
  matchType: "exact" | "normalized" | "numeric" | "partial" | "none";
}

export interface ChallengeScore {
  challengeId: string;
  answerAccuracy: number;      // correct / total
  confidenceCalibration: number; // how well confidence predicts correctness
  citationGroundingScore: number;
  trapEvasionRate: number;
  passed: boolean;             // AA >= 0.6
  questionScores: QuestionScore[];
}

export interface SplitSummary {
  answerAccuracy: number;
  meanConfidence: number;
  trapEvasionRate: number;
  passRate: number;
  challengeCount: number;
  questionCount: number;
}

export interface FormatBreakdown {
  count: number;      // Number of challenges parsed with this format
  accuracy: number;   // Accuracy for challenges using this format
  total: number;      // Total questions from this format
}

export interface BenchmarkResults {
  model: string;
  benchmarkVersion: string;
  timestamp: string;
  summary: {
    answerAccuracy: number;
    confidenceWeightedAccuracy: number;  // accuracy weighted by confidence
    confidenceCalibration: number;       // ECE (expected calibration error)
    meanConfidence: number;
    trapEvasionRate: number;
    citationGroundingScore: number;
    passRate: number;
    challengesEvaluated: number;
    questionsEvaluated: number;
    formatFailureRate: number;
  };
  byCategory: Record<QuestionCategory, { accuracy: number; meanConfidence: number; count: number }>;
  byDomain: Record<string, { accuracy: number; meanConfidence: number; count: number }>;
  byHops: Record<string, { accuracy: number; meanConfidence: number; count: number }>;
  byDifficulty: Record<string, { accuracy: number; meanConfidence: number; count: number }>;
  bySplit?: Record<BenchmarkSplit, SplitSummary>;
  byFormat?: Record<ParseFormat, FormatBreakdown>; // Breakdown by response format
  transferGap?: number; // |AA_real - AA_synthetic| — lower means synthetic predicts real well
  challengeScores: ChallengeScore[];
}

// ── Benchmark Dataset (top-level container) ──

export interface BenchmarkDataset {
  name: string;
  version: string;
  createdAt: string;
  description: string;
  challenges: BenchmarkChallenge[];
  domainBreakdown: Record<string, number>;
  categoryBreakdown: Record<QuestionCategory, number>;
  splitBreakdown: Record<BenchmarkSplit, number>;
  stats: {
    totalChallenges: number;
    totalQuestions: number;
    totalTraps: number;
    verificationPassRate: number; // fraction that passed all verification
    domainCount: number;
  };
}
