import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import Anthropic from '@anthropic-ai/sdk';
import { l10n } from '../i18n';
import { Profile } from '../models/profile';

export type SpeedTestStatus = 'success' | 'error';

export interface SpeedTestResult {
  profile: Profile;
  status: SpeedTestStatus;
  durationMs: number;
  firstTokenMs?: number;
  speedTokensPerSec?: number;
  model?: string;
  error?: string;
}

const DEFAULT_TIMEOUT_MS = 15_000;
const DEFAULT_CONCURRENCY = 3;
const USER_SETTINGS_PATH = path.join(os.homedir(), '.claude', 'settings.json');

interface ClaudeUserSettings {
  model?: unknown;
  env?: Record<string, unknown>;
}

interface ResolvedSpeedConfig {
  model?: string;
  token?: string;
  baseURL?: string;
}

export class SpeedTester {
  /**
   * 生成随机测速提示，避免请求模式过于固定被封禁
   */
  private generateRandomPrompt(): string {
    // 多种问题模板
    const templates: ((n: number) => string)[] = [
      (n: number) => `List the numbers from 1 to ${n}, one per line.`,
      (n: number) => `Count from 1 to ${n}, each number on a new line.`,
      (n: number) => `Please output numbers 1 through ${n}, line by line.`,
      (n: number) => `Write numbers from 1 up to ${n}, one per line.`,
      (n: number) => `Enumerate numbers 1 to ${n}, each on its own line.`,
      // 字母表类
      (n: number) => `List the first ${n} letters of the alphabet, one per line.`,
      (n: number) => `Write letters A through ${String.fromCharCode(64 + n)}, one per line.`,
      // 简单任务类
      (n: number) => `Say "hello world" ${n} times, each on a new line.`,
      (n: number) => `Print the word "hello world" ${n} times, one per line.`,
      (n: number) => `List ${n} common colors, one per line.`,
      (n: number) => `Name ${n} fruits, one per line.`,
    ];

    // 随机选择模板和参数
    const template = templates[Math.floor(Math.random() * templates.length)];
    // 随机生成参数：字母类用 10-26，数字类用 20-50
    const randomNum = Math.floor(Math.random() * 31) + 20; // 20-50

    return template(randomNum);
  }

  async listModels(baseURL?: string, token?: string, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<string[]> {
    const normalizedBaseURL = this.normalizeModelsBaseURL(baseURL);
    if (!normalizedBaseURL) {
      throw new Error(l10n('webviewBaseUrlRequired'));
    }

    const client = this.createClient(
      {
        token: this.trimString(token),
        baseURL: normalizedBaseURL,
      },
      timeoutMs,
    );

    const ids: string[] = [];
    for await (const model of client.models.list()) {
      if (typeof model.id === 'string' && model.id.trim()) {
        ids.push(model.id.trim());
      }
    }

    return [...new Set(ids)].sort((a, b) => a.localeCompare(b));
  }

  async testProfile(profile: Profile, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<SpeedTestResult> {
    const startedAt = Date.now();
    const config = this.resolveConfig(profile);
    const model = config.model;

    if (!model) {
      return {
        profile,
        status: 'error',
        durationMs: 0,
        error: l10n('speedMissingModel'),
      };
    }

    try {
      const client = this.createClient(config, timeoutMs);

      let firstTokenAt = 0;
      let outputTokens = 0;

      const stream = await client.messages.create({
        model,
        max_tokens: 128,
        messages: [{ role: 'user', content: this.generateRandomPrompt() }],
        stream: true,
      });

      for await (const event of stream) {
        if (event.type === 'content_block_delta') {
          if (!firstTokenAt) {
            firstTokenAt = Date.now();
          }
        }
        if (event.type === 'message_delta') {
          if (event.usage && event.usage.output_tokens) {
            outputTokens = event.usage.output_tokens;
          }
        }
      }

      const endedAt = Date.now();
      const firstTokenMs = firstTokenAt ? (firstTokenAt - startedAt) : undefined;
      const durationMs = endedAt - startedAt;

      const generationDurationMs = endedAt - (firstTokenAt || startedAt);
      const speedTokensPerSec = generationDurationMs > 0 && outputTokens > 1
        ? Number(((outputTokens - 1) / (generationDurationMs / 1000)).toFixed(1))
        : undefined;

      return {
        profile,
        status: 'success',
        durationMs,
        firstTokenMs,
        speedTokensPerSec,
        model,
      };
    } catch (err) {
      return {
        profile,
        status: 'error',
        durationMs: Date.now() - startedAt,
        model,
        error: this.formatError(err),
      };
    }
  }

  async testProfiles(
    profiles: Profile[],
    onResult: (result: SpeedTestResult, completed: number, total: number) => void,
    concurrency = DEFAULT_CONCURRENCY,
  ): Promise<SpeedTestResult[]> {
    const results: SpeedTestResult[] = [];
    let nextIndex = 0;
    let completed = 0;

    const worker = async () => {
      while (nextIndex < profiles.length) {
        const profile = profiles[nextIndex++];
        if (!profile) continue;

        const result = await this.testProfile(profile);
        results.push(result);
        completed++;
        onResult(result, completed, profiles.length);
      }
    };

    const workerCount = Math.min(concurrency, profiles.length);
    await Promise.all(Array.from({ length: workerCount }, worker));
    return results;
  }

  private resolveConfig(profile: Profile): ResolvedSpeedConfig {
    const userSettings = this.readClaudeUserSettings();
    const userEnv = userSettings?.env;
    const token = this.trimString(profile.env.ANTHROPIC_AUTH_TOKEN)
      || this.trimString(userEnv?.ANTHROPIC_AUTH_TOKEN)
      || this.trimString(userEnv?.ANTHROPIC_API_KEY)
      || undefined;

    return {
      model: this.stripOneMillionContextSuffix(
        this.trimString(profile.env.ANTHROPIC_MODEL)
        || this.trimString(profile.env.ANTHROPIC_DEFAULT_OPUS_MODEL)
        || this.trimString(profile.env.ANTHROPIC_DEFAULT_SONNET_MODEL)
        || this.trimString(profile.env.ANTHROPIC_DEFAULT_HAIKU_MODEL)
        || undefined,
      ),
      token,
      baseURL: this.trimString(profile.env.ANTHROPIC_BASE_URL)
        || this.trimString(userEnv?.ANTHROPIC_BASE_URL)
        || undefined,
    };
  }

  private createClient(config: ResolvedSpeedConfig, timeoutMs: number): Anthropic {
    const baseURL = this.normalizeBaseURL(config.baseURL);
    const authOptions = this.resolveAuthOptions(config.token);

    return new Anthropic({
      ...authOptions,
      baseURL,
      timeout: timeoutMs,
      maxRetries: 0,
      defaultHeaders: config.token ? undefined : { 'X-Api-Key': null, Authorization: null },
    });
  }

  private resolveAuthOptions(token?: string): { apiKey: string | null; authToken: string | null } {
    if (!token) return { apiKey: null, authToken: null };
    return token.startsWith('sk-ant-')
      ? { apiKey: token, authToken: null }
      : { apiKey: null, authToken: token };
  }

  private readClaudeUserSettings(): ClaudeUserSettings | undefined {
    if (!fs.existsSync(USER_SETTINGS_PATH)) return undefined;

    try {
      const settings = JSON.parse(fs.readFileSync(USER_SETTINGS_PATH, 'utf-8')) as ClaudeUserSettings;
      return settings && typeof settings === 'object' ? settings : undefined;
    } catch {
      return undefined;
    }
  }

  private trimString(value: unknown): string | undefined {
    if (typeof value !== 'string') return undefined;
    const trimmed = value.trim();
    return trimmed || undefined;
  }

  private stripOneMillionContextSuffix(value?: string): string | undefined {
    if (!value) return undefined;
    return value.endsWith('[1m]') ? value.slice(0, -4).trim() : value;
  }

  private normalizeBaseURL(baseURL?: string): string | undefined {
    const trimmed = baseURL?.trim();
    if (!trimmed) return undefined;
    return trimmed.replace(/\/+$/, '').replace(/\/v1$/, '');
  }

  private normalizeModelsBaseURL(baseURL?: string): string | undefined {
    const normalized = this.normalizeBaseURL(baseURL);
    return normalized?.replace(/\/anthropic$/i, '');
  }

  public formatError(error: unknown): string {
    if (error instanceof Anthropic.AuthenticationError) {
      return l10n('speedAuthFailed');
    }
    if (error instanceof Anthropic.PermissionDeniedError) {
      return l10n('speedPermissionDenied');
    }
    if (error instanceof Anthropic.NotFoundError) {
      return l10n('speedModelOrEndpointNotFound');
    }
    if (error instanceof Anthropic.RateLimitError) {
      return l10n('speedRateLimited');
    }
    if (error instanceof Anthropic.APIError) {
      return l10n('speedApiError', error.status ? ` ${error.status}` : '', error.message);
    }
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }
}
