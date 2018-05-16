package positivenews.android.POJO;

public class News {

    private String image_url;


    private String news_url;
    private String news_title;
    private String news_summary;
    private String news_date;

    public News(String image_url, String news_url, String news_title, String news_summary, String news_date) {
        this.image_url = image_url;
        this.news_url = news_url;
        this.news_title = news_title;
        this.news_summary = news_summary;
        this.news_date = news_date;
    }

    public News(String image_url, String news_title, String news_summary, String news_url) {
        this.image_url = image_url;
        this.news_title = news_title;
        this.news_summary = news_summary;
        this.news_url = news_url;
    }

    public News(String lastUpdateDate) {
        this.news_date = lastUpdateDate;
    }

    public String getImage_url() {
        return image_url;
    }

    public String getNews_url() {
        return news_url;
    }

    public String getNews_title() {
        return news_title;
    }

    public String getNews_summary() {
        return news_summary;
    }

    public String getNews_date() {
        return news_date;
    }

    public void setImage_url(String image_url) {
        this.image_url = image_url;
    }

    public void setNews_url(String news_url) {
        this.news_url = news_url;
    }

    public void setNews_title(String news_title) {
        this.news_title = news_title;
    }

    public void setNews_summary(String news_summary) {
        this.news_summary = news_summary;
    }

    public void setNews_date(String news_date) {
        this.news_date = news_date;
    }

}
