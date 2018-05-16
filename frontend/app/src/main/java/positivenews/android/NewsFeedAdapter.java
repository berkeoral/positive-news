package positivenews.android;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.GestureDetector;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.squareup.picasso.Picasso;

import java.util.ArrayList;
import java.util.List;

import positivenews.android.POJO.News;

public class NewsFeedAdapter extends RecyclerView.Adapter<NewsFeedAdapter.ViewHolder> {

    private ArrayList<News> newsArrayList;
    private Context context;

    public NewsFeedAdapter(ArrayList<News> data, Context context) {
        this.context = context;
        if (data != null) {
            newsArrayList = data;
        }
        else{
            newsArrayList = new ArrayList<News>();
        }
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        public TextView title;
        public TextView summary;
        public ImageView thumbnail;

        public ViewHolder(View newsCard) {
            super(newsCard);
            this.title = newsCard.findViewById(R.id.newsFeedNewsTitleID);
            this.summary = newsCard.findViewById(R.id.newsFeedNewsSummaryID);
            this.thumbnail = newsCard.findViewById(R.id.newsFeedNewsThumbnailID);

            newsCard.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent browserIntent = new Intent(Intent.ACTION_VIEW,
                            Uri.parse(newsArrayList.get(getLayoutPosition()).getNews_url()));
                    context.startActivity(browserIntent);
                }
            });
        }
    }

    @Override
    public NewsFeedAdapter.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(parent.getContext());
        View newsView = inflater.inflate(R.layout.news_card_layout, parent, false);

        return new NewsFeedAdapter.ViewHolder(newsView);
    }

    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {
        holder.title.setText(newsArrayList.get(position).getNews_title());
        holder.summary.setText(newsArrayList.get(position).getNews_summary());
        loadThumbnail(newsArrayList.get(position).getImage_url(), holder.thumbnail);
    }

    public void addItem(News news) {
        newsArrayList.add(news);
        notifyItemInserted(newsArrayList.size() - 1);
    }

    @Override
    public int getItemCount() {
        if (newsArrayList == null) {
            return 0;
        } else {
            return newsArrayList.size();
        }
    }


    private void loadThumbnail(String url, ImageView imageView) {
        Picasso.get().load(url)
                .resize((int) context.getResources().getDimension(R.dimen.thumbnail_width),
                        (int) context.getResources().getDimension(R.dimen.thumbnail_height))
                .centerCrop()
                .into(imageView);

    }

}