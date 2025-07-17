import { describe, test } from 'vitest';
import assert from 'node:assert';
import { Ajv, ValidateFunction } from 'ajv';
import { ChatCompletionResponse } from '../src/lib/schema_types.ts';
import response_schema from '../../schemas/chatcompletionresponse_schema.json';
import message_schema from '../../schemas/chatmessage_schema.json';
import frontend_config from '../../config_module/config_frontend.json';

const BASEURL: string = frontend_config['backend_base_url'];
const ajv: Ajv = new Ajv();
const response_validator: ValidateFunction<ChatCompletionResponse> = ajv
	.addSchema(message_schema)
	.compile<ChatCompletionResponse>(response_schema);

/*
NOTE: these test cases are meant as sanity checks for the testing environment
if any of these fail, the code below probably needs to be refactored
*/
describe('environment', () => {
	// this is the base URL for the mock backend, so it's important to verify it's correct
	test('document.baseURI is localhost', () => {
		assert.equal(document.baseURI, 'http://localhost:3000/');
	});
	test('is running in node.js', () => {
		// c.f. https://stackoverflow.com/questions/4224606/how-to-check-whether-a-script-is-running-under-node-js
		assert.strictEqual(typeof process, 'object');
		assert.strictEqual(process + '', '[object process]');
	});
	test('is running with jsdom', () => {
		// c.f. https://github.com/jsdom/jsdom/issues/1537
		assert(navigator.userAgent.includes('jsdom'));
	});
});

describe('GET /vfm-mock', () => {
	test('accepts GET', async () => {
		const response: Response = await fetch(`${BASEURL}/vfm-mock`, { method: 'GET' });
		assert.strictEqual(response.status, 200);
		assert.strictEqual(await response.text(), 'lorem ipsum dolor sit amet');
	});
});

describe('POST /v1/chat/completions', () => {
	// PARTITION:
	// - method: GET (shouldn't work), POST (should work)
	// - mandatory parameters: model, messages
	// - optional parameters: stream, temperature, thread_id
	test("doesn't accept GET", async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'GET'
		});
		assert.strictEqual(response.status, 405); // NOTE: HTTP 405 = "method not allowed"
	});

	test("doesn't accept POST, wrong Content-Type", async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'text/plain' }
		});
		assert.strictEqual(response.status, 400); // NOTE: HTTP 400 = "bad request"
		assert.strictEqual(response.headers.get('Content-Type'), 'text/plain');
	});

	test("doesn't accept POST, no body", async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' }
		});
		assert.strictEqual(response.status, 400);
		assert.strictEqual(response.headers.get('Content-Type'), 'text/plain');
	});

	test("doesn't accept POST, wrong body type", async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(42)
		});
		assert.strictEqual(response.status, 400);
		assert.strictEqual(response.headers.get('Content-Type'), 'text/plain');
	});

	test("doesn't accept POST, empty body", async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({})
		});
		assert.strictEqual(response.status, 400);
		assert.strictEqual(response.headers.get('Content-Type'), 'text/plain');
	});

	test("doesn't accept POST, model, no messages", async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ model: 'ark-reason' })
		});
		assert.strictEqual(response.status, 400);
		assert.strictEqual(response.headers.get('Content-Type'), 'text/plain');
	});

	test("doesn't accept POST, no model, messages", async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ messages: [{ role: 'user', content: 'hello world' }] })
		});
		assert.strictEqual(response.status, 400);
		assert.strictEqual(response.headers.get('Content-Type'), 'text/plain');
	});

	test('POST with model and messages', async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model: 'ark-reason',
				messages: [{ role: 'user', content: 'hello world' }]
			})
		});
		assert.strictEqual(response.status, 200);
		const responseJSON: unknown = await response.json();
		assert(response_validator(responseJSON));
	});

	test('POST with model, messages, temperature', async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model: 'ark-reason',
				messages: [{ role: 'user', content: 'hello world' }],
				temperature: 2
			})
		});
		assert.strictEqual(response.status, 200);
		const responseJSON: unknown = await response.json();
		assert(response_validator(responseJSON));
	});

	test('POST with model, messages, thread_id', async () => {
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model: 'ark-reason',
				messages: [{ role: 'user', content: 'hello world' }],
				thread_id: 'abc123'
			})
		});
		assert.strictEqual(response.status, 200);
		const responseJSON: unknown = await response.json();
		assert(response_validator(responseJSON));
	});

	test.skip('POST with model, messages, stream'); // TODO: implement this test once I figure out how to implement streaming
});
