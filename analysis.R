library(readr)
library(dplyr)
library(stringr)
library(jsonlite)
library(plotly)
library(tm)
filepath <- paste0(getwd(), "/naive_result/result0.json")
review_lines <- read_lines(filepath)
reviews_combined <- str_c("[", str_c(review_lines, collapse = ", "), "]")
reviews <- fromJSON(reviews_combined) %>% flatten() %>% tbl_df()

result <-reviews
result$stars <- factor(result$stars)

median(result$stars)
table(result$stars)
(result$positive_probability)
# 5 is the most common rating.
# mean rating is 3.6
# median is 4
# mean positive probability is 0.53

#exploratory analysis
star1 <- subset(result, stars == 1)
mean(star1$positive_probability) # 0.031
median(star1$positive_probability) # 1.901e-07

star2 <- subset(result, stars == 2)
mean(star2$positive_probability) # 0.136
median(star2$positive_probability) # 0.0004


star3 <- subset(result, stars == 3)
mean(star3$positive_probability) # 0.416
median(star3$positive_probability) # 0.240

star4 <- subset(result, stars == 4)
mean(star4$positive_probability) # 0.696
median(star4$positive_probability) # 0.942

star5 <- subset(result, stars == 5)
mean(star5$positive_probability) # 0.775
median(star5$positive_probability) # 0.977

#box plot
plot_ly(data = result, y = ~positive_probability, color = ~stars, type = 'box')

#outliers= 2 standard deviations away
# star2:
star2 <- subset(star2, positive_probability > 0 & positive_probability < 0.6)
star2_lower = mean(star2$positive_probability) - sd(star2$positive_probability) * 3
star2_upper = mean(star2$positive_probability) + sd(star2$positive_probability) * 3
star2_OL <- subset(star2, positive_probability < star2_lower | positive_probability > star2_upper)
star2_not_OL <- subset(star2, positive_probability > star2_lower & positive_probability < star2_upper)
# star3:
star3 <- subset(star3, positive_probability > 0 & positive_probability < 0.8)
star3_lower = mean(star3$positive_probability) - sd(star3$positive_probability) * 2
star3_upper = mean(star3$positive_probability) + sd(star3$positive_probability) * 2
star3_OL <- subset(star3, positive_probability < star3_lower | positive_probability > star3_upper)
star3_not_OL <- subset(star3, positive_probability > star3_lower & positive_probability < star3_upper)
# star4:
star4 <- subset(star4, positive_probability > 0.4)
star4_lower = mean(star4$positive_probability) - sd(star4$positive_probability) * 3
star4_upper = mean(star4$positive_probability) + sd(star4$positive_probability) * 3
star4_OL <- subset(star4, positive_probability < star4_lower | positive_probability > star4_upper)
star4_not_OL <- subset(star4, positive_probability > star4_lower & positive_probability < star4_upper)


