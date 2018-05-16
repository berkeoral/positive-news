package positivenews.android;

import java.util.List;

import positivenews.android.POJO.News;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface NewsFeedInterface {
    @POST("news/fetch")
    Call<List<News>> fetchNews(@Body News news);
}
