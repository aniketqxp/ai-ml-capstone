import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.resolve(
  HERE,
  '../../public/evaluation-v2/en_CA_Banking_1586889.json',
);
const CALL_PATH = '/calls/en_CA_Banking_1586889';

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
    category: 'required_control',
    category_label: 'Required step',
    title: 'Identity verification was not demonstrated',
    summary:
      'The call moved into account activity without a grounded verification exchange.',
    evidence_ids: ['evidence-control-identity'],
    counter_evidence_count: 0,
    reliability_note: null,
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
    },
  ];
  run.presentation.additional_positive_count = 0;
  run.presentation.recommended_action = {
    action_type: 'create_review_case',
    execution: 'automatic',
    label: 'Create review case',
    execution_label: 'Automatic',
    basis_finding_ids: [finding.finding_id],
    automation_allowed: true,
    requires_human_approval: false,
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

test('renders the truthful incomplete shadow state', async ({ page }) => {
  const errors = captureBrowserErrors(page);
  await page.goto(CALL_PATH);
  await expect(
    page.getByText('Evaluation incomplete', { exact: true }),
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

test('renders attention, evidence, action, and local feedback', async ({
  page,
}) => {
  const errors = captureBrowserErrors(page);
  await page.route('**/evaluation-v2/en_CA_Banking_1586889.json', (route) =>
    route.fulfill({ json: attentionFixture() }),
  );
  await page.route('**/calls/en_CA_Banking_1586889/feedback', (route) =>
    route.fulfill({ json: {} }),
  );
  await page.goto(CALL_PATH);
  await expect(page.getByText('Needs attention', { exact: true })).toBeVisible();
  await expect(
    page.getByRole('heading', {
      name: 'Identity verification was not demonstrated',
    }),
  ).toBeVisible();
  await expect(page.getByText('Handled well')).toBeVisible();
  await expect(page.getByText('Key evidence')).toBeVisible();
  await page.getByRole('button', { name: 'Mark action complete' }).click();
  await expect(
    page.getByText('Feedback saved in this local preview.'),
  ).toBeVisible();
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

test('keeps unsupported domains on the v1 page', async ({ page }) => {
  await page.goto('/calls/en_CA_Health_1587315');
  await expect(
    page.getByRole('heading', { name: 'Call Compliance Player' }),
  ).toBeVisible();
});
