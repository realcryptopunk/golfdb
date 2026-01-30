import Foundation
import UIKit

// MARK: - Response Models

struct SwingAnalysisResponse: Codable {
    let events: [SwingEvent]
    let swingMetrics: SwingMetrics?
    let videoInfo: VideoInfo

    enum CodingKeys: String, CodingKey {
        case events
        case swingMetrics = "swing_metrics"
        case videoInfo = "video_info"
    }
}

struct SwingEvent: Codable, Identifiable {
    let eventId: Int
    let eventName: String
    let frame: Int
    let timeSeconds: Double
    let confidence: Double
    var frameImageBase64: String?  // Only present with /analyze-with-frames

    var id: Int { eventId }

    enum CodingKeys: String, CodingKey {
        case eventId = "event_id"
        case eventName = "event_name"
        case frame
        case timeSeconds = "time_seconds"
        case confidence
        case frameImageBase64 = "frame_image_base64"
    }

    /// Decode base64 image if available
    var frameImage: UIImage? {
        guard let base64 = frameImageBase64,
              let data = Data(base64Encoded: base64) else { return nil }
        return UIImage(data: data)
    }

    /// Confidence as percentage string
    var confidencePercent: String {
        String(format: "%.1f%%", confidence * 100)
    }
}

struct SwingMetrics: Codable {
    let backswingDuration: Double
    let downswingDuration: Double
    let totalSwingDuration: Double
    let averageConfidence: Double

    enum CodingKeys: String, CodingKey {
        case backswingDuration = "backswing_duration"
        case downswingDuration = "downswing_duration"
        case totalSwingDuration = "total_swing_duration"
        case averageConfidence = "average_confidence"
    }
}

struct VideoInfo: Codable {
    let fps: Double
    let frameCount: Int
    let durationSeconds: Double

    enum CodingKeys: String, CodingKey {
        case fps
        case frameCount = "frame_count"
        case durationSeconds = "duration_seconds"
    }
}

struct APIError: Codable {
    let error: String
}

// MARK: - API Client

class GolfSwingAPIClient {

    static let shared = GolfSwingAPIClient()

    // MARK: - Configuration

    /// Base URL for the API server
    /// Change this to your production server URL
    var baseURL: String = "http://localhost:8080"

    /// Request timeout in seconds
    var timeoutInterval: TimeInterval = 120

    private init() {}

    // MARK: - Public Methods

    /// Analyze a golf swing video
    /// - Parameters:
    ///   - videoURL: Local file URL of the video
    ///   - includeFrames: If true, returns base64-encoded frame images for each event
    ///   - completion: Callback with result
    func analyzeSwing(
        videoURL: URL,
        includeFrames: Bool = false,
        completion: @escaping (Result<SwingAnalysisResponse, Error>) -> Void
    ) {
        let endpoint = includeFrames ? "/analyze-with-frames" : "/analyze"
        let url = URL(string: baseURL + endpoint)!

        // Create multipart form request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = timeoutInterval

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        // Build multipart body
        var body = Data()

        // Add video file
        let filename = videoURL.lastPathComponent
        let mimeType = "video/mp4"

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"video\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)

        do {
            let videoData = try Data(contentsOf: videoURL)
            body.append(videoData)
        } catch {
            completion(.failure(error))
            return
        }

        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        // Make request
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }

            guard let data = data else {
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GolfSwingAPI", code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                }
                return
            }

            // Check for API error
            if let apiError = try? JSONDecoder().decode(APIError.self, from: data) {
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GolfSwingAPI", code: -2,
                        userInfo: [NSLocalizedDescriptionKey: apiError.error])))
                }
                return
            }

            // Decode response
            do {
                let response = try JSONDecoder().decode(SwingAnalysisResponse.self, from: data)
                DispatchQueue.main.async {
                    completion(.success(response))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }

        task.resume()
    }

    /// Analyze a golf swing video using async/await
    @available(iOS 15.0, *)
    func analyzeSwing(videoURL: URL, includeFrames: Bool = false) async throws -> SwingAnalysisResponse {
        return try await withCheckedThrowingContinuation { continuation in
            analyzeSwing(videoURL: videoURL, includeFrames: includeFrames) { result in
                switch result {
                case .success(let response):
                    continuation.resume(returning: response)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Check if the API server is healthy
    func checkHealth(completion: @escaping (Bool) -> Void) {
        let url = URL(string: baseURL + "/health")!

        URLSession.shared.dataTask(with: url) { data, response, error in
            let isHealthy = error == nil && (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async {
                completion(isHealthy)
            }
        }.resume()
    }
}

// MARK: - SwiftUI Example View

import SwiftUI

@available(iOS 15.0, *)
struct SwingAnalysisView: View {
    @State private var analysisResult: SwingAnalysisResponse?
    @State private var isAnalyzing = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {

                    // Analysis Button
                    Button(action: {
                        // In real app, use video picker
                        analyzeVideo()
                    }) {
                        HStack {
                            Image(systemName: "video.fill")
                            Text(isAnalyzing ? "Analyzing..." : "Select Video to Analyze")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isAnalyzing ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(isAnalyzing)
                    .padding(.horizontal)

                    // Error Message
                    if let error = errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .padding()
                    }

                    // Results
                    if let result = analysisResult {

                        // Swing Metrics Card
                        if let metrics = result.swingMetrics {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Swing Metrics")
                                    .font(.headline)

                                HStack {
                                    MetricView(title: "Backswing", value: String(format: "%.2fs", metrics.backswingDuration))
                                    MetricView(title: "Downswing", value: String(format: "%.2fs", metrics.downswingDuration))
                                    MetricView(title: "Total", value: String(format: "%.2fs", metrics.totalSwingDuration))
                                }

                                Text("Avg Confidence: \(String(format: "%.1f%%", metrics.averageConfidence * 100))")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                            .padding(.horizontal)
                        }

                        // Events List
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Detected Events")
                                .font(.headline)
                                .padding(.horizontal)

                            ForEach(result.events) { event in
                                EventRow(event: event)
                            }
                        }
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Golf Swing Analysis")
        }
    }

    func analyzeVideo() {
        // Example: In real app, use UIImagePickerController or PhotosPicker
        // For demo, assume video URL exists
        guard let videoURL = Bundle.main.url(forResource: "sample_swing", withExtension: "mp4") else {
            errorMessage = "No sample video found"
            return
        }

        isAnalyzing = true
        errorMessage = nil

        Task {
            do {
                let result = try await GolfSwingAPIClient.shared.analyzeSwing(
                    videoURL: videoURL,
                    includeFrames: true
                )
                analysisResult = result
            } catch {
                errorMessage = error.localizedDescription
            }
            isAnalyzing = false
        }
    }
}

@available(iOS 15.0, *)
struct MetricView: View {
    let title: String
    let value: String

    var body: some View {
        VStack {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

@available(iOS 15.0, *)
struct EventRow: View {
    let event: SwingEvent

    var confidenceColor: Color {
        if event.confidence > 0.8 { return .green }
        if event.confidence > 0.5 { return .yellow }
        return .red
    }

    var body: some View {
        HStack {
            // Event frame image (if available)
            if let image = event.frameImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 80, height: 60)
                    .cornerRadius(8)
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: 80, height: 60)
                    .cornerRadius(8)
                    .overlay(
                        Text("\(event.frame)")
                            .font(.caption)
                    )
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(event.eventName)
                    .font(.headline)

                Text(String(format: "%.3fs", event.timeSeconds))
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Confidence indicator
            Text(event.confidencePercent)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(confidenceColor)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(confidenceColor.opacity(0.2))
                .cornerRadius(4)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
    }
}
