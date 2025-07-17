// NOTE: replace `localhost:3000` by actual domain name once we get one
import { Ajv, ValidateFunction } from 'ajv';
import { ChatCompletionRequest } from '../src/lib/schema_types.ts';
import request_schema from '../../schemas/chatcompletionrequest_schema.json';
import message_schema from '../../schemas/chatmessage_schema.json';
import frontend_config from '../../config_module/config_frontend.json';

const BASEURL: string = frontend_config['backend_base_url'];
const ajv: Ajv = new Ajv();
// NOTE: you MUST type `request_validator` this way to narrow the type of `reqJSON`, otherwise a priori TypeScript just thinks this is a generic `ValidateFunction` object
const request_validator: ValidateFunction<ChatCompletionRequest> = ajv
	.addSchema(message_schema)
	.compile<ChatCompletionRequest>(request_schema);

/**
 * Handles *only* requests to POST /v1/chat/completions
 * NOTE: this is exported so we can use `expect().toHaveBeenCalled()` when spying on it during testing
 *
 * @param req the HTTP request to handle
 * @returns the HTTP response to send
 */
export async function handleChatCompletions(req: Request): Promise<Response> {
	// check HTTP method
	if (req.method !== 'POST') {
		return new Response('wrong HTTP method!', {
			status: 405,
			headers: { 'Content-Type': 'text/plain' }
		});
	}
	// check Content-Type
	if (req.headers.get('Content-Type') !== 'application/json') {
		return new Response('wrong Content-Type', {
			status: 400,
			headers: { 'Content-Type': 'text/plain' }
		});
	}
	// try to parse JSON
	let reqJSON: unknown;
	try {
		reqJSON = await req.json();
	} catch {
		return new Response('no or malformed body', {
			status: 400,
			headers: { 'Content-Type': 'text/plain' }
		});
	}
	// input validation for `reqJSON`
	if (!request_validator(reqJSON)) {
		return new Response(`malformed request: ${request_validator.errors}`, {
			status: 400,
			headers: { 'Content-Type': 'text/plain' }
		});
	}

	if (reqJSON['stream'] === true) {
		// TODO: implement streaming later
		return new Response('no streaming yet', { status: 501 });
	}

	return new Response(
		JSON.stringify({
			id: 'abcdefghijklmnopqrstuvwxyz', // randomize?
			object: 'chat.completion',
			created: ~~(Date.now() / 1000), // round so schema's correct; c.f. https://stackoverflow.com/questions/4228356/how-to-perform-an-integer-division-and-separately-get-the-remainder-in-javascr
			model: 'example',
			choices: [
				{
					index: 0,
					message: {
						role: 'assistant',
						content: 'lorem ipsum' // randomize?
					},
					finish_reason: null
				}
			]
		}),
		{ status: 200 }
	);
}

/**
 * Handles all HTTP requests to the mock backend
 * @param req the HTTP request to handle
 * @returns the HTTP response to send
 */
export async function handleMockBackendRequest(req: Request): Promise<Response> {
	if (req.url === `${BASEURL}/vfm-mock`) {
		return new Response('lorem ipsum dolor sit amet', { status: 200 });
	} else if (req.url === `${BASEURL}/v1/chat/completions`) {
		return await handleChatCompletions(req);
	} else {
		return new Response('not found', { status: 404 });
	}
}
