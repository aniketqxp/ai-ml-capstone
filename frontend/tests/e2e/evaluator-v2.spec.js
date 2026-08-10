import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.resolve(
  HERE,
  '../../public/evaluation-v2/en_CA_Banking_1586889.json',
);
const SENTIMENT_FIXTURE = path.resolve(
  HERE,
  '../../../ml-services/outputs/backend/sentiment_calls_with_features/banking/en_CA_Banking_1586889_backend_sentiment_with_features.json',
);
const CALL_PATH = '/calls/en_CA_Banking_1586889';

test.beforeEach(async ({ page }) => {
  const sentiment = JSON.parse(fs.readFileSync(SENTIMENT_FIXTURE, 'utf-8'));
  await page.route('**/sentiment/en_CA_Banking_1586889.json', (route) =>
    route.fulfill({ json: sentiment }),
  );
});

function captureBrowserErrors(page) {
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

function attentionFixture() {
  const run = JSON.parse(fs.readFileSync(FIXTURE, 'utf-8'));
  const finding = {
    finding_id: 'finding-control-identity',
    category: 'process',
    category_label: 'Call handling',
    title: 'Transfer amount confirmed incorrectly',
    summary:
      'The agent repeated a different amount than the customer requested.',
    evidence_ids: ['evidence-control-identity'],
    counter_evidence_count: 0,
    reliability_note: null,
    outcome: 'incorrect',
  };
  run.decision_sha256 = 'a'.repeat(64);
  run.presentation.decision_sha256 = run.decision_sha256;
  run.presentation.state = 'needs_attention';
  run.presentation.evaluation_status = 'complete';
  run.presentation.attention_required = true;
  run.presentation.headline = 'Needs attention';
  run.presentation.summary =
    'One evidence-backed finding requires review.';
  run.presentation.primary_reasons = [finding];
  run.presentation.additional_reason_count = 0;
  run.presentation.positive_highlights = [
    {
      finding_id: 'finding-intent-confirmed',
      category: 'process',
      category_label: 'Call handling',
      title: 'Customer request was confirmed',
      summary:
        'The agent restated the requested transfer before proceeding.',
      evidence_ids: ['evidence-intent-confirmed'],
      counter_evidence_count: 0,
      reliability_note: null,
      outcome: 'effective',
    },
  ];
  run.presentation.manager_questions = [
    {
      question_id: 'call.objective',
      question: 'Did the agent understand and address the objective?',
      answer: 'partly',
      answer_label: 'Partly',
      summary: finding.summary,
      evidence_ids: finding.evidence_ids,
    },
    {
      question_id: 'call.workflow',
      question: 'Did the agent follow the applicable workflow?',
      answer: 'no',
      answer_label: 'No',
      summary: finding.summary,
      evidence_ids: finding.evidence_ids,
    },
    {
      question_id: 'call.friction',
      question: 'Did the interaction introduce customer friction?',
      answer: 'unclear',
      answer_label: 'Unable to determine',
      summary: '2 customer segments matched a high-precision explicit-text rule.',
      evidence_ids: [],
    },
    {
      question_id: 'call.follow_up',
      question: 'Is follow-up, coaching, or customer action required?',
      answer: 'yes',
      answer_label: 'Yes',
      summary: 'Email manager review summary',
      evidence_ids: finding.evidence_ids,
    },
  ];
  run.presentation.checklist = [
    {
      requirement_id: 'transfer.amount_confirmed',
      title: finding.title,
      category: finding.category,
      status: 'incorrect',
      summary: finding.summary,
      evidence_ids: finding.evidence_ids,
      promoted: true,
    },
  ];
  run.presentation.acoustic_context = {
    status: 'limited',
    coverage_label: 'Audio support on 4 of 8 segments',
    conclusion: 'No sustained vocal-friction pattern crossed the provisional support gate.',
    observations: [],
  };
  run.presentation.additional_positive_count = 0;
  run.presentation.recommended_action = {
    action_type: 'manager_review',
    execution: 'automatic',
    label: 'Email manager review summary',
    execution_label: 'Automatic',
    basis_finding_ids: [finding.finding_id],
    automation_allowed: true,
    requires_human_approval: false,
    delivery: 'email',
    audience: 'manager',
  };
  run.presentation.evidence = [
    {
      evidence_id: 'evidence-control-identity',
      finding_ids: [finding.finding_id],
      purposes: ['reason'],
      kind: 'transcript',
      speaker: 'agent',
      start_seconds: 40.49,
      end_seconds: 48.56,
      text: 'Before we start, can I verify your identity?',
      seekable: true,
      supporting_only: false,
    },
    {
      evidence_id: 'evidence-intent-confirmed',
      finding_ids: ['finding-intent-confirmed'],
      purposes: ['positive'],
      kind: 'transcript',
      speaker: 'agent',
      start_seconds: 54.26,
      end_seconds: 61.0,
      text: 'You would like to set up an automatic transfer, correct?',
      seekable: true,
      supporting_only: false,
    },
  ];
  run.presentation.completeness_notice = null;
  run.presentation.details.additional_findings = [];
  run.presentation.details.additional_positive_findings = [];
  run.presentation.details.evidence = [];
  return run;
}

test('renders a complete evidence-gated evaluation', async ({ page }) => {
  const errors = captureBrowserErrors(page);
  await page.goto(CALL_PATH);
  await expect(
    page.getByText('No review finding identified', { exact: true }),
  ).toBeVisible();
  await expect(page.getByText('Effective moments')).toBeVisible();
  await expect(page.getByText('Call assessment')).toBeVisible();
  await expect(page.getByText('Applicable checks')).toBeVisible();
  await expect(page.getByText('Voice & sentiment')).toBeVisible();
  await expect(page.getByText('Pauses', { exact: true })).toBeVisible();
  await expect(page.getByText('13.3%', { exact: true })).toBeVisible();
  expect(
    await page.locator('.turn-bubble').first()
      .locator('[data-sentiment="positive"]').count(),
  ).toBeGreaterThan(1);
  await expect(page.getByText('positive', { exact: true })).toHaveCount(0);
  await expect(
    page.getByRole('navigation', { name: 'Call chapters' }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Conversation' })).toBeVisible();
  await expect(page.getByText('Compliance score')).toHaveCount(0);
  await expect(page.getByText('Workflow score')).toHaveCount(0);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  expect(errors).toEqual([]);
  await page.screenshot({
    path: 'test-results/evaluator-v2-desktop.png',
    fullPage: true,
  });
});

test('renders layered attention, evidence, and email action', async ({
  page,
}) => {
  const errors = captureBrowserErrors(page);
  await page.route('**/evaluation-v2/en_CA_Banking_1586889.json', (route) =>
    route.fulfill({ json: attentionFixture() }),
  );
  await page.route('**/calls/en_CA_Banking_1586889/feedback', (route) =>
    route.fulfill({ json: {} }),
  );
  await page.route('**/calls/en_CA_Banking_1586889/email', async (route) => {
    if (route.request().method() === 'POST') {
      await route.fulfill({
        json: {
          status: 'sent',
          audience: 'manager',
          recipient: 'manager@example.com',
          subject: 'Manager review',
          sent_at: '2026-08-10T12:00:00',
        },
      });
      return;
    }
    await route.fulfill({ json: { notifications: [] } });
  });
  await page.goto(CALL_PATH);
  await expect(page.getByText('Needs attention', { exact: true })).toBeVisible();
  await expect(
    page.getByText('One evidence-backed finding requires review.', { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText(
      '2 customer segments matched a high-precision explicit-text rule.',
      { exact: true },
    ),
  ).toHaveCount(0);
  await expect(
    page.getByRole('heading', {
      name: 'Transfer amount confirmed incorrectly',
    }),
  ).toBeVisible();
  await expect(page.getByText('Incorrect handling')).toBeVisible();
  await expect(page.getByText('Effective moments')).toBeVisible();
  const emailTrigger = page.getByRole('button', { name: 'Open email action' });
  await expect(emailTrigger).toBeVisible();
  await expect(page.getByRole('dialog', { name: 'Email action' })).toHaveCount(0);
  await emailTrigger.click();
  await expect(page.getByRole('dialog', { name: 'Email action' })).toBeVisible();
  await expect(
    page.getByText('Email manager review summary').last(),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Send email' }).click();
  await expect(page.getByText('Email sent', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('restores word-level playback highlighting', async ({ page }) => {
  const errors = captureBrowserErrors(page);
  await page.goto(CALL_PATH);
  const firstTimedWord = page.locator('[data-word-start]').first();
  await expect(firstTimedWord).toBeVisible();
  const start = Number(await firstTimedWord.getAttribute('data-word-start'));
  await page.locator('input[aria-label="Seek through call"]').evaluate(
    (element, value) => {
      element.value = String(value);
      element.dispatchEvent(new Event('input', { bubbles: true }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
    },
    start + 0.01,
  );
  await expect(firstTimedWord).toHaveAttribute('data-active', 'true');
  expect(errors).toEqual([]);
});

test('fits the v2 call page on a mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route('**/evaluation-v2/en_CA_Banking_1586889.json', (route) =>
    route.fulfill({ json: attentionFixture() }),
  );
  await page.goto(CALL_PATH);
  await expect(page.getByText('Needs attention', { exact: true })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({
    path: 'test-results/evaluator-v2-mobile.png',
    fullPage: true,
  });
});

test('keeps unsupported domains in the evaluator v2 shell', async ({ page }) => {
  await page.goto('/calls/en_CA_Health_1587315');
  await expect(
    page.getByText('Evaluation unavailable', { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole('heading', { name: 'Call Compliance Player' }),
  ).toHaveCount(0);
});
