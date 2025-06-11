Các tác giả đã thực hiện đề tài nghiên cứu của mình theo một quy trình bao gồm hai phần chính: **tạo văn bản tiết lộ nội dung clickbait** và **phân loại loại tiết lộ nội dung clickbait**. Dưới đây là chi tiết quy trình:

**1. Chuẩn bị Dữ liệu:**
*   Các tác giả đã sử dụng **tập dữ liệu SemEval-2023**, có sẵn trên Zenodo, để điều chỉnh mô hình GPT-2 và huấn luyện các bộ phân loại học máy SBERT.
*   Tập dữ liệu này chứa **4.000 bài viết clickbait**, được phân loại vào ba loại tiết lộ nội dung clickbait riêng biệt: **ngắn (phrase)**, **đoạn (passage)**, và **đa đoạn (multipart)**. Tập dữ liệu cũng bao gồm một cột chứa các tiết lộ nội dung clickbait đã được con người chú thích để tham chiếu.
*   **3.200 mẫu** được sử dụng để huấn luyện, trong khi **800 mẫu còn lại** được dùng để kiểm tra.

**2. Mô hình Tạo văn bản Tiết lộ Nội dung Clickbait (Spoiler Generation Model):**
*   **Mô hình được sử dụng:** Các tác giả đã sử dụng **mô hình GPT-2 medium** đã được huấn luyện trước. Họ cũng thử nghiệm với mô hình T5 nhưng GPT-2 medium đã tinh chỉnh cho kết quả tốt hơn với yêu cầu tính toán thấp hơn.
*   **Dữ liệu đầu vào:** Để tạo văn bản tiết lộ, mô hình được cung cấp một cột kết hợp của **`postText`** (nội dung bài đăng) và **`targetParagraph`** (đoạn văn mục tiêu) từ tập dữ liệu SemEval 2023.
*   **Mục tiêu:** Mục tiêu là tạo ra văn bản tương tự như nội dung của cột `Spoiler` (tiết lộ) có sẵn trong tập dữ liệu.
*   **Tiền xử lý và Huấn luyện:**
    *   Văn bản kết hợp được **tokenize** bằng cách sử dụng bộ mã hóa dựa trên transformer trước khi đưa vào mô hình GPT-2.
    *   Mô hình GPT-2 được huấn luyện trước bằng cách sử dụng **mô hình ngôn ngữ nhân quả** (casual language modeling), nơi nó dự đoán phân phối xác suất của token tiếp theo.
    *   Quá trình **tinh chỉnh** (fine-tuning) được thực hiện bằng cách sử dụng **hàm mất mát entropy chéo phân loại xoắn ốc** (spiral categorical cross-entropy loss).
    *   Các **siêu tham số** như tốc độ học, kích thước lô và số epoch huấn luyện được hiệu chỉnh cẩn thận. Mô hình được tinh chỉnh qua nhiều epoch và mô hình có hiệu suất tốt nhất trên tập xác thực được giữ lại.
*   **Đánh giá hiệu suất:**
    *   Khả năng dịch của mô hình được đánh giá trên một tập kiểm tra độc lập bằng cách sử dụng các chỉ số như **BLEU** (Bilingual Evaluation Understudy), **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation), **BERTScore**, và **METEOR**.
    *   Mô hình GPT-2 medium đã tinh chỉnh đạt **điểm BLEU là 0.58**, **ROUGE là 0.68**, **BERTScore là 0.46**, và **METEOR là 0.47**, vượt trội so với các mô hình tiên tiến trước đây và T5.

**3. Mô hình Phân loại Loại Tiết lộ Nội dung Clickbait (Spoiler Type Classification Model):**
*   **Mục tiêu:** Phân loại các tiết lộ nội dung clickbait thành ba loại: **ngắn (phrase)**, **đoạn (passage)** và **đa đoạn (multi)**.
*   **Trích xuất đặc trưng (Feature Extraction):**
    *   Một **bộ chuyển đổi câu (sentence transformer)**, cụ thể là **SBERT**, được sử dụng để tạo ra vector nhúng cho mỗi câu.
    *   Năm cột dữ liệu được sử dụng để tạo vector nhúng: **`postText`**, **`targetTitle`**, **`targetParagraph`**, **`targetDescription`**, và **`targetKeywords`**.
    *   Các vector nhúng được trích xuất từ mỗi câu trong năm cột này được **nối lại** để tạo thành một vector nhúng có kích thước 1920 (mỗi cột đóng góp một vector kích thước 384).
*   **Phân loại (Classification):**
    *   Ma trận nhúng đã tạo được đưa vào **tám bộ phân loại học máy khác nhau** để xác định loại tiết lộ.
    *   Trong số các bộ phân loại được thử nghiệm (bao gồm Naive Bayes, KNN, Decision Tree, AdaBoost, Logistic Regression, Gradient Boosting, Random Forest), **mô hình Máy vectơ hỗ trợ (SVM)** cho thấy hiệu suất vượt trội.
    *   Các tác giả cũng đã thử nghiệm với các kỹ thuật nhúng TF-IDF và GloVe nhưng **SBERT** đã cho kết quả tốt hơn đáng kể.
*   **Đánh giá hiệu suất:**
    *   Hiệu suất của mô hình phân loại được đánh giá bằng các chỉ số: **độ đúng (accuracy)**, **độ chính xác (precision)**, **độ nhạy (recall)**, **điểm F1 (F1-score)**, và **Hệ số tương quan Matthews (MCC)**.
    *   Mô hình **SBERT+SVM** đề xuất đã đạt được độ chính xác 0.83, độ nhạy 0.82, điểm F1 0.80, và MCC 0.63. Hiệu suất này đã **vượt trội** so với các mô hình tiên tiến trước đây, bao gồm cả những mô hình sử dụng nhúng RoBERTa và DistilBERT.

Tóm lại, quy trình nghiên cứu bao gồm việc **sử dụng dữ liệu SemEval-2023**, **tinh chỉnh mô hình GPT-2 medium để tạo văn bản tiết lộ** từ `postText` và `targetParagraph`, và **phát triển một khung phân loại loại tiết lộ** bằng cách sử dụng nhúng SBERT từ nhiều thành phần văn bản khác nhau và bộ phân loại SVM.