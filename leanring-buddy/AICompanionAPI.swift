//
//  AICompanionAPI.swift
//  leanring-buddy
//
//  Abstraction over different AI vision backends (Claude, OpenAI, etc.).
//  CompanionManager depends on this protocol rather than a concrete class,
//  so the AI provider can be swapped at runtime based on user model selection.
//

import Foundation

/// Protocol that all AI companion backends must conform to.
/// Defines the streaming vision interface CompanionManager uses for voice responses.
protocol AICompanionAPI {
    func analyzeImageStreaming(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)],
        userPrompt: String,
        onTextChunk: @MainActor @Sendable (String) -> Void
    ) async throws -> (text: String, duration: TimeInterval)
}
