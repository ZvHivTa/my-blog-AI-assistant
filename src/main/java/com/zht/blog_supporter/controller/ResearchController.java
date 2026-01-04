package com.zht.blog_supporter.controller;

import com.google.genai.Client;
import com.google.genai.types.GenerateContentResponse;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.InputStreamResource;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/research")
public class ResearchController {

    private final VectorStore vectorStore;

    // 注入 API Key
    @Value("${spring.ai.openai.api-key}")
    private String googleApiKey;

    public ResearchController(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    // 1. 上传接口 (保持不变，依然使用本地 Embedding)
    @PostMapping("/upload")
    public String uploadPaper(@RequestParam("file") MultipartFile file) throws IOException {
        TikaDocumentReader reader = new TikaDocumentReader(new InputStreamResource(file.getInputStream()));
        List<Document> documents = reader.get();
        TokenTextSplitter splitter = new TokenTextSplitter();
        List<Document> splitDocuments = splitter.apply(documents);
        vectorStore.add(splitDocuments);
        return "上传成功！已处理文献：" + file.getOriginalFilename() + "，片段数：" + splitDocuments.size();
    }

    // 2. 聊天接口 (使用 Google 官方 SDK)
    @GetMapping("/chat")
    public String chat(@RequestParam String question) {
        // A. RAG 搜库
        List<Document> similarDocs = vectorStore.similaritySearch(
                SearchRequest.query(question).withTopK(2)
        );

        String context = similarDocs.isEmpty() ? "无相关资料" :
                similarDocs.stream().map(Document::getContent).collect(Collectors.joining("\n\n"));

        String prompt = "你是一个科研助手。请根据以下已知信息回答问题。\n\n" +
                "已知信息：\n" + context + "\n\n" +
                "用户问题：" + question;

        try {
            // B. 初始化官方 Client (非常简单!)
            Client client = Client.builder()
                    .apiKey(googleApiKey)
                    .build();

            // C. 调用 generateContent
            // 这里可以直接用 "gemini-2.0-flash-exp" 或者 "gemini-1.5-flash"
            GenerateContentResponse response = client.models.generateContent(
                    "gemini-flash-latest", // 或者用 "gemini-1.5-flash"
                    prompt,
                    null // 第三个参数是 config，暂时传 null 即可
            );

            // D. 直接获取文本 (SDK 自动处理了 JSON 解析)
            return response.text();

        } catch (Exception e) {
            e.printStackTrace();
            return "调用 Google SDK 失败: " + e.getMessage();
        }
    }
}