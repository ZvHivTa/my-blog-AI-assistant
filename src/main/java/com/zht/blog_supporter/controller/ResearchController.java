package com.zht.blog_supporter.controller;

// 移除官方 SDK 的 import
// import com.google.genai.Client; ...

import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/research")
public class ResearchController {

    private final VectorStore vectorStore;
    private final RestClient restClient; // ✅ 加回来

    @Value("${spring.ai.openai.api-key}")
    private String googleApiKey;

    public ResearchController(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
        this.restClient = RestClient.builder().build(); // ✅ 加回来
    }

    // 1. 上传接口 (保持不变)
    @PostMapping("/upload")
    public String uploadPaper(@RequestParam("file") MultipartFile file) throws IOException {
        TikaDocumentReader reader = new TikaDocumentReader(new InputStreamResource(file.getInputStream()));
        List<Document> documents = reader.get();
        TokenTextSplitter splitter = new TokenTextSplitter();
        List<Document> splitDocuments = splitter.apply(documents);
        vectorStore.add(splitDocuments);
        return "上传成功！已处理文献：" + file.getOriginalFilename() + "，片段数：" + splitDocuments.size();
    }

    // 2. 聊天接口 (通过 Cloudflare 代理调用)
    @GetMapping("/chat")
    public String chat(@RequestParam String question) {
        // A. RAG 搜库
        List<Document> similarDocs = vectorStore.similaritySearch(
                SearchRequest.query(question).withTopK(2)
        );
        String context = similarDocs.isEmpty() ? "无相关资料" :
                similarDocs.stream().map(Document::getContent).collect(Collectors.joining("\n\n"));

        String systemPrompt = "你是一个科研助手。请根据已知信息回答问题。";
        String userPrompt = "已知信息：\n" + context + "\n\n问题：" + question;

        // 使用你的 Cloudflare Worker 地址！
        // 格式：https://你的worker地址/v1beta/models/gemini-2.0-flash-exp:generateContent...
        // ⚠️ 把下面的域名换成你刚才在 Cloudflare 申请到的！
        String cfWorkerUrl = "https://broken-night-61cf.zvhivta.workers.dev";

        String url = cfWorkerUrl + "/v1beta/models/gemini-2.0-flash-lite-001:generateContent?key=" + googleApiKey.trim();

        Map<String, Object> requestBody = Map.of(
                "contents", List.of(
                        Map.of("parts", List.of(
                                Map.of("text", systemPrompt + "\n" + userPrompt)
                        ))
                )
        );

        try {
            // 发送请求给 Cloudflare
            String rawJson = restClient.post()
                    .uri(url)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(requestBody)
                    .retrieve()
                    .body(String.class);

            // 解析 JSON
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(rawJson);
            String answer = root.path("candidates").get(0)
                    .path("content").path("parts").get(0)
                    .path("text").asText();
            return answer;

        } catch (Exception e) {
            e.printStackTrace();
            return "调用失败 (Cloudflare): " + e.getMessage();
        }
    }
}