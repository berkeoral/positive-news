package positivenews.android;

import android.content.SharedPreferences;
import android.support.v4.widget.SwipeRefreshLayout;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.widget.Toast;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

import positivenews.android.POJO.News;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class PositiveNews extends AppCompatActivity {

    private SwipeRefreshLayout newsFeedRL;
    private RecyclerView newsFeedRV;
    private NewsFeedAdapter newsFeedRVAdapter;
    private LinearLayoutManager newsFeedRVLayoutManager;

    private Retrofit retrofit;
    private Gson gson;
    private NewsFeedInterface newsFeedInterface;

    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_positive_news);

        prefs = getSharedPreferences(
                getResources().getString(R.string.sp_name), MODE_PRIVATE);

        gson = new GsonBuilder()
                .setLenient()
                .create();
        retrofit = new Retrofit.Builder()
                .baseUrl(getString(R.string.web_service))
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build();
        newsFeedInterface = retrofit.create(NewsFeedInterface.class);

        newsFeedRV = findViewById(R.id.news_feed_rv);
        newsFeedRVAdapter = new NewsFeedAdapter(null, this);
        newsFeedRVLayoutManager = new LinearLayoutManager(this);
        newsFeedRVLayoutManager.setReverseLayout(true);
        newsFeedRVLayoutManager.setStackFromEnd(true);
        newsFeedRV.setLayoutManager(newsFeedRVLayoutManager);
        newsFeedRV.setAdapter(newsFeedRVAdapter);

        newsFeedRL = findViewById(R.id.newsFeedRefreshLayout);
        newsFeedRL.setOnRefreshListener(
                new SwipeRefreshLayout.OnRefreshListener() {
                    @Override
                    public void onRefresh() {
                        fetchNews();
                    }
                }
        );

        // TODO add sql database
        SharedPreferences.Editor editor = getSharedPreferences(
                getResources().getString(R.string.sp_name), MODE_PRIVATE).edit();
        editor.putString(getResources().getString(R.string.sp_update_date),
                getResources().getString(R.string.base_update_date));
        editor.apply();

        fetchNews();
    }

    private void fetchNews() {
        final ArrayList<News> fetchedNews = new ArrayList<>();
        String lastUpdateDate = prefs.getString(getResources().getString(R.string.sp_update_date)
                , getResources().getString(R.string.base_update_date));
        News requestBody = new News(lastUpdateDate);

        Call<List<News>> call = newsFeedInterface.fetchNews(requestBody);

        call.enqueue(new Callback<List<News>>() {
            @Override
            public void onResponse(Call<List<News>> call, Response<List<News>> response) {
                if (response.isSuccessful()) {
                    fetchedNews.addAll(response.body());
                    if (fetchedNews.size() != 0) {
                        for (News article : fetchedNews) {
                            newsFeedRVAdapter.addItem(article);
                        }
                        updateLastUpdatedDate();
                    }
                }
                newsFeedRL.setRefreshing(false);
                newsFeedRV.scrollToPosition(newsFeedRVAdapter.getItemCount() - 1);
            }

            @Override
            public void onFailure(Call<List<News>> call, Throwable t) {
                Toast.makeText(PositiveNews.this,
                        getResources().getText(R.string.connection_error), Toast.LENGTH_SHORT).show();
                newsFeedRL.setRefreshing(false);
            }
        });
    }

    private void updateLastUpdatedDate() {
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        SharedPreferences.Editor editor = getSharedPreferences(
                getResources().getString(R.string.sp_name), MODE_PRIVATE).edit();
        editor.putString(getResources().getString(R.string.sp_update_date), timestamp.toString());
        editor.apply();
    }

}
