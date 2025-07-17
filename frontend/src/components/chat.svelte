<script lang="ts">
	import { Ajv, type ValidateFunction } from 'ajv';
	import {
		type ChatMessage,
		type ChatCompletionResponse,
		type ChatChoice
	} from '../lib/schema_types.js';
	import response_schema from '../../../schemas/chatcompletionresponse_schema.json';
	import message_schema from '../../../schemas/chatmessage_schema.json';
	import frontend_config from '../../../config_module/config_frontend.json';

	const BASEURL: string = frontend_config['backend_base_url'];
	const ajv: Ajv = new Ajv();
	const response_validator: ValidateFunction<ChatCompletionResponse> = ajv
		.addSchema(message_schema)
		.compile<ChatCompletionResponse>(response_schema);

	let currentMessages: Array<ChatMessage> = $state([
		{
			role: 'assistant',
			content: "Hello! I'm your calendar assistant. How can I help you today?"
		}
	]);
	let currentUserMessage: string = $state('');

	// c.f. https://lightningchart.com/js-charts/api-documentation/v5.2.0/types/MouseEventHandler.html for typing
	async function handleMessageSending(): Promise<void> {
		currentMessages = [...currentMessages, { role: 'user', content: currentUserMessage }]; // reassign for better reactivity
		const response: Response = await fetch(`${BASEURL}/v1/chat/completions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model: 'ark-reason',
				messages: currentMessages
			})
		});
		const responseJSON: unknown = await response.json();
		if (!response_validator(responseJSON)) {
			// if invalid, log and also remove the last message for now
			// TODO: make this more user-friendly and also figure out how to test this
			console.log('response JSON was invalid!');
			currentMessages = currentMessages.slice(0, -2);
			return;
		}
		// pick an arbitrary choice (TODO: randomize or show a picker UI explicitly?)
		const responseChoices: Array<ChatChoice> = responseJSON.choices;
		const myMessage: ChatMessage | undefined = responseChoices.at(0)?.message;
		if (myMessage === undefined) {
			// TODO: make this more user-friendly?
			console.log('there are no choices in the response!');
			currentMessages = currentMessages.slice(0, -2);
			return;
		}
		// if it's valid, add!
		currentMessages = [...currentMessages, myMessage];

		// reset the input
		currentUserMessage = '';
	}

	async function handleKeydown(event: KeyboardEvent): Promise<void> {
		if (event.key === 'Enter') {
			event.preventDefault();
			await handleMessageSending();
		}
	}
</script>

<div class="chat-container" data-testid="chat-container">
	<div class="chat-messages" data-testid="chat-messages">
		<!-- NOTE: for testing purposes, messages are numbered in `data-testid` using zero-indexing -->
		{#each currentMessages.entries() as [index, message] (index)}
			<div class={`message ${message.role}`} data-testid={`message${index}`}>
				<p>{message.content}</p>
			</div>
		{/each}
	</div>
	<div class="chat-input-container" data-testid="chat-input-container">
		<input
			type="text"
			class="chat-input"
			id="chatInput"
			placeholder="Type a message..."
			bind:value={currentUserMessage}
			onkeydown={handleKeydown}
		/>
		<!-- c.f. https://svelte.dev/tutorial/svelte/text-inputs -->
		<button class="send-button" id="sendButton" onclick={handleMessageSending}>Send</button>
	</div>
</div>
