"""Generate a deterministic ~500 KB Vietnamese Wikipedia-like fixture file for local testing."""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

BASE_ARTICLES: list[tuple[str, list[str]]] = [
    (
        "Hà Nội",
        [
            "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam và là một trong hai đô thị đặc biệt của cả nước. Nằm ở vùng đồng bằng sông Hồng, thành phố giữ vai trò trung tâm chính trị, hành chính, văn hóa, giáo dục và khoa học công nghệ quan trọng bậc nhất.",
            "Lịch sử của Hà Nội gắn liền với kinh thành Thăng Long do vua Lý Thái Tổ dời đô từ Hoa Lư ra năm 1010. Trải qua nhiều triều đại Lý, Trần, Lê, Nguyễn cùng các biến cố lịch sử hiện đại, thành phố vẫn lưu giữ được hệ thống di sản kiến trúc, phố cổ và hồ tự nhiên độc đáo.",
            "Về địa lý tự nhiên, sông Hồng chảy qua khu vực trung tâm và bồi đắp phù sa màu mỡ cho vùng đồng bằng xung quanh. Khí hậu Hà Nội mang đặc trưng nhiệt đới gió mùa ẩm với bốn mùa xuân, hạ, thu, đông phân hóa khá rõ rệt.",
        ],
    ),
    (
        "Thành phố Hồ Chí Minh",
        [
            "Thành phố Hồ Chí Minh là trung tâm kinh tế, tài chính, thương mại và dịch vụ lớn nhất của Việt Nam. Đô thị này nằm ở miền Đông Nam Bộ, tiếp giáp với các tỉnh Bình Dương, Đồng Nai, Tây Ninh, Long An, Bà Rịa - Vũng Tàu và Tiền Giang.",
            "Hệ thống sông Sài Gòn và sông Đồng Nai đóng vai trò huyết mạch trong giao thông thủy, cấp nước sinh hoạt và phát triển cảng biển. Nhiều khu công nghiệp, khu chế xuất và khu công nghệ cao tập trung tại đây đã thu hút lực lượng lao động đông đảo từ khắp cả nước.",
            "Đời sống văn hóa và ẩm thực của thành phố thể hiện sự giao thoa cởi mở giữa truyền thống Nam Bộ với nhiều vùng miền và quốc tế. Các công trình lịch sử như Dinh Độc Lập, Nhà thờ Đức Bà, Bưu điện Trung tâm và Chợ Bến Thành là những điểm đến quen thuộc.",
        ],
    ),
    (
        "Trí tuệ nhân tạo",
        [
            "Trí tuệ nhân tạo là một nhánh của khoa học máy tính nghiên cứu cách xây dựng các hệ thống có khả năng thực hiện những nhiệm vụ đòi hỏi trí thông minh của con người như nhận dạng tiếng nói, thị giác máy tính, suy luận logic và xử lý ngôn ngữ tự nhiên.",
            "Từ thập niên 2010, sự kết hợp giữa dữ liệu quy mô lớn, thuật toán học sâu và năng lực tính toán song song trên bộ xử lý đồ họa đã tạo ra bước nhảy vọt về độ chính xác. Kiến trúc Transformer ra đời năm 2017 mở đường cho các mô hình ngôn ngữ lớn hiện đại.",
            "Bên cạnh lợi ích về năng suất lao động và tự động hóa, cộng đồng nghiên cứu cũng chú trọng đến tính giải thích được, sự công bằng, an toàn dữ liệu và mức tiêu thụ năng lượng của các hệ thống học máy.",
        ],
    ),
    (
        "Sông Mê Kông",
        [
            "Sông Mê Kông là một trong những con sông lớn nhất thế giới, bắt nguồn từ cao nguyên Tây Tạng và chảy qua lãnh thổ Trung Quốc, Myanmar, Lào, Thái Lan, Campuchia trước khi đổ ra Biển Đông qua hệ thống chín cửa sông tại đồng bằng sông Cửu Long.",
            "Tại Việt Nam, sông được gọi là sông Cửu Long với hai nhánh chính là sông Tiền và sông Hậu. Nguồn nước và lượng phù sa dồi dào từ thượng nguồn tạo nên vùng sản xuất lúa gạo, cây ăn trái và thủy sản trọng điểm của quốc gia.",
            "Những năm gần đây, biến đổi khí hậu, nước biển dâng và các công trình thủy điện trên dòng chính đặt ra thách thức lớn đối với cân bằng sinh thái và sinh kế của hàng triệu cư dân lưu vực.",
        ],
    ),
    (
        "Văn học Việt Nam",
        [
            "Văn học Việt Nam bao gồm hai bộ phận gắn bó mật thiết là văn học dân gian và văn học viết. Kho tàng truyền thuyết, cổ tích, ca dao, tục ngữ phản ánh sinh động đời sống lao động, khát vọng công lý và tâm hồn của người dân qua nhiều thế hệ.",
            "Trong giai đoạn trung đại, văn học viết sử dụng cả chữ Hán và chữ Nôm, đạt đến đỉnh cao với những tác phẩm như Bình Ngô đại cáo của Nguyễn Trãi hay Truyện Kiều của đại thi hào Nguyễn Du.",
            "Bước sang thế kỷ XX, chữ Quốc ngữ phổ biến rộng rãi đã thúc đẩy sự phát triển mạnh mẽ của thơ mới, tiểu thuyết hiện đại, truyện ngắn và phê bình văn học, góp phần hình thành diện mạo văn xuôi đương đại.",
        ],
    ),
    (
        "Vật lý lượng tử",
        [
            "Cơ học lượng tử là lý thuyết vật lý cơ bản mô tả hành vi của vật chất và năng lượng ở thang kích thước nguyên tử và hạ nguyên tử. Khác với cơ học cổ điển, nhiều đại lượng vật lý như năng lượng hay mô men động lượng chỉ nhận các giá trị rời rạc.",
            "Những khái niệm cốt lõi bao gồm lưỡng tính sóng hạt, nguyên lý bất định Heisenberg và sự rối lượng tử. Phương trình Schrödinger cho phép tính toán hàm sóng mô tả phân bố xác suất tìm thấy hạt trong không gian theo thời gian.",
            "Các ứng dụng thực tiễn của vật lý lượng tử hiện diện khắp đời sống hiện đại, từ linh kiện bán dẫn, laser, chụp cộng hưởng từ y khoa cho tới công nghệ máy tính lượng tử đang được nghiên cứu.",
        ],
    ),
    (
        "Ẩm thực Huế",
        [
            "Ẩm thực Huế nổi tiếng bởi sự tinh tế trong cách phối hợp gia vị, màu sắc hài hòa và quy trình chế biến công phu. Vùng đất cố đô từng là nơi hội tụ nghệ nhân từ nhiều miền để phục vụ triều đình nhà Nguyễn cũng như đời sống dân gian.",
            "Những món ăn tiêu biểu có thể kể đến bún bò Huế với hương sả và mắm ruốc đậm đà, cơm hến thanh mát, cùng hàng loạt loại bánh dân gian như bánh bèo, bánh nậm, bánh lọc, bánh ram ít.",
            "Ngoài hương vị, người Huế còn coi trọng nghệ thuật trình bày và lễ nghi thưởng thức, biến mỗi bữa ăn thành một nét văn hóa đặc sắc khó trộn lẫn.",
        ],
    ),
    (
        "Vịnh Hạ Long",
        [
            "Vịnh Hạ Long thuộc tỉnh Quảng Ninh nằm ở bờ tây vịnh Bắc Bộ, nổi bật với gần hai nghìn hòn đảo đá vôi lớn nhỏ nhô lên trên làn nước xanh ngọc. Quá trình kiến tạo địa chất hàng trăm triệu năm đã tạo ra hệ thống hang động karst kỳ vĩ như hang Sửng Sốt, động Thiên Cung.",
            "Nhờ giá trị ngoại hạng về cảnh quan tự nhiên và địa chất địa mạo, vịnh Hạ Long đã được UNESCO nhiều lần ghi danh là Di sản Thiên nhiên Thế giới. Khu vực này cũng là môi trường sống của nhiều loài sinh vật biển và thực vật đặc hữu.",
            "Công tác quản lý du lịch bền vững và bảo vệ môi trường nước luôn được đặt lên hàng đầu nhằm gìn giữ di sản trước áp lực phát triển kinh tế biển.",
        ],
    ),
    (
        "Toán học rời rạc",
        [
            "Toán học rời rạc nghiên cứu các cấu trúc toán học mang tính rời rạc thay vì liên tục, bao gồm lý thuyết tập hợp, logic mệnh đề, tổ hợp, lý thuyết đồ thị và số học đồng dư.",
            "Trong khoa học máy tính, đồ thị và cây là công cụ nền tảng để mô hình hóa mạng lưới giao thông, mạng xã hội, tối ưu hóa đường đi ngắn nhất và phân tích độ phức tạp thuật toán.",
            "Các khái niệm xác suất rời rạc và lý thuyết thông tin như entropy Shannon hay mã hóa nén dữ liệu đóng vai trò trực tiếp trong thiết kế mô hình ngôn ngữ và truyền thông số.",
        ],
    ),
    (
        "Hệ sinh thái rừng ngập mặn",
        [
            "Rừng ngập mặn phát triển dọc theo các vùng cửa sông, bãi bồi ven biển nhiệt đới và cận nhiệt đới nơi có thủy triều lên xuống hằng ngày. Những loài cây tiêu biểu như đước, mắm, vẹt sở hữu hệ rễ chống hoặc rễ thở thích nghi tốt với nền đất bùn thiếu oxy.",
            "Thảm thực vật này đóng vai trò như bức tường sinh học tự nhiên giúp chắn sóng, giảm xói lở bờ biển và hấp thụ lượng lớn khí các-bon. Đồng thời, đây còn là vườn ươm sinh sản cho nhiều loài tôm, cua, cá và chim nước di cư.",
            "Tại Việt Nam, rừng ngập mặn Cần Giờ và bán đảo Cà Mau là những khu dự trữ sinh quyển tiêu biểu đang được phục hồi và bảo vệ nghiêm ngặt.",
        ],
    ),
]

SUBTOPICS = [
    "Lịch sử hình thành",
    "Đặc điểm nổi bật",
    "Phân loại và cấu trúc",
    "Vai trò trong thực tiễn",
    "Nghiên cứu và phát triển",
    "Ảnh hưởng văn hóa và xã hội",
]

REGIONS = [
    "miền Bắc",
    "miền Trung",
    "miền Nam",
    "đồng bằng sông Hồng",
    "duyên hải Nam Trung Bộ",
    "Tây Nguyên",
    "đồng bằng sông Cửu Long",
]

EXTRA_SENTENCES = [
    "Nhiều tài liệu chuyên khảo đã ghi nhận sự thay đổi rõ rệt theo từng giai đoạn phát triển từ thế kỷ XIX đến nay.",
    "Các số liệu thống kê thực địa cho thấy tính đa dạng cao về thành phần, quy mô cũng như mối liên hệ giữa các yếu tố tự nhiên và con người.",
    "Giới nghiên cứu thường sử dụng phương pháp định lượng kết hợp khảo sát dài hạn để kiểm chứng giả thuyết ban đầu.",
    "Trong bối cảnh hội nhập quốc tế, việc chuẩn hóa dữ liệu và bảo tồn tri thức bản địa ngày càng nhận được sự quan tâm rộng rãi.",
    "Các chương trình đào tạo đại học và viện nghiên cứu đóng góp phần quan trọng vào việc nâng cao chất lượng nguồn nhân lực chuyên môn.",
    "Hiệu quả triển khai thực tế phụ thuộc vào điều kiện hạ tầng, nguồn lực tài chính và sự phối hợp giữa nhiều cơ quan liên quan.",
]


def build_fixture_records(target_bytes: int = 510_000, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    records: list[dict[str, str]] = []
    total_bytes = 0
    idx = 0

    # Add explicit edge-case documents first so fixtures always cover all test scenarios.
    edge_cases = [
        # 1. NFD decomposed + invisible controls + trailing Wikipedia sections
        (
            "\ufeff"
            + unicodedata.normalize(
                "NFD",
                BASE_ARTICLES[0][1][0] + "\n\n" + BASE_ARTICLES[0][1][1],
            ).replace(" ", " \u200b ")
            + "\n\nXem thêm\n* Kinh thành Thăng Long\nTham khảo\n1. Đại Việt sử ký toàn thư"
        ),
        # 2. Markup leftovers + literal <|endoftext|> + emoji
        (
            BASE_ARTICLES[2][1][0]
            + '<ref name="ref1">Trích dẫn mẫu</ref><!-- ghi chú ẩn --> '
            + "Chuỗi đặc biệt <|endoftext|> xuất hiện tự nhiên trong văn bản và biểu tượng 🚀.\n"
            + '{| class="wikitable"\n|-\n| Cột 1 || Cột 2\n|}\n\n'
            + BASE_ARTICLES[2][1][1]
            + "\n\nChú thích\n1. Tài liệu kiểm thử"
        ),
        # 3. Stub (< 250 chars) - should be filtered out
        "Đây là một bài viết sơ khai quá ngắn chỉ có một câu.",
        # 4. Non-Latin spam - should be filtered out
        "これは日本語のテストデータです。" * 25,
        # 5. Pure numbers / table - should be filtered out
        "1024 2048 4096 8192 16384 32768 " * 20,
    ]
    for doc in edge_cases:
        line = json.dumps({"text": doc}, ensure_ascii=False)
        records.append({"text": doc})
        total_bytes += len(line.encode("utf-8")) + 1

    # Add an exact duplicate of a valid long doc to exercise dedup_exact.
    dup_doc = "\n\n".join(BASE_ARTICLES[1][1])
    for _ in range(2):
        line = json.dumps({"text": dup_doc}, ensure_ascii=False)
        records.append({"text": dup_doc})
        total_bytes += len(line.encode("utf-8")) + 1

    while total_bytes < target_bytes:
        title, paras = BASE_ARTICLES[idx % len(BASE_ARTICLES)]
        subtopic = SUBTOPICS[idx % len(SUBTOPICS)]
        region = REGIONS[(idx * 3) % len(REGIONS)]
        year = 1950 + (idx * 7) % 75
        code = 100 + idx

        selected_paras = list(paras)
        rng.shuffle(selected_paras)
        extras = rng.sample(EXTRA_SENTENCES, k=3)
        body = (
            f"{title} — {subtopic} (mục {code})\n\n"
            + f"{selected_paras[0]} Tại khu vực {region}, nhiều khảo sát công bố vào năm {year} đã làm rõ bối cảnh hình thành.\n\n"
            + f"{selected_paras[1]} {extras[0]} {extras[1]}\n\n"
            + f"{selected_paras[2]} {extras[2]}"
        )
        if idx % 5 == 0:
            body += "\n\nXem thêm\n* Danh mục liên quan\nTham khảo\n1. Tư liệu tham khảo"
        if idx % 7 == 0:
            body = unicodedata.normalize("NFD", body)

        line = json.dumps({"text": body}, ensure_ascii=False)
        records.append({"text": body})
        total_bytes += len(line.encode("utf-8")) + 1
        idx += 1

    return records


def main() -> None:
    out_path = Path("fixtures/sample_vi_wiki.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = build_fixture_records()
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"Generated {len(records)} fixture records at {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
