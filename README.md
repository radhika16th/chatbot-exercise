# POC: AI Chatbot for Exercise Recommendations

## Problem Statement

People experiencing body pain often lack quick access to exercise suggestions that are trustworthy. This chatbot will help with that by providing basic advice in a conversational format.

## Goal

To build a chatbot using Gemini or an open-source LLM that suggests safe and recommended exercises based on the symptoms described by the user.

## Tools

Language Model: Google Gemini or HuggingFace (Mistral?)
Interface: Flask or FastAPI
Prompt Safety: System prompts for safety
Hosting: Local env

## Steps

1. Research Gemini API and other alternative LLMs that can be used
2. Create prompts for pain and exercise suggestions 
3. Build the interface (on console first, then GUI)
4. Test some common queries
5. Disclaimers

## Challenges

Making sure that the suggestions are medically safe
Avoid hallucinations
API rate limits and token costs when using Gemini

## Outcome

A functional chatbot that showcases how LLMs can help with wellness use cases
